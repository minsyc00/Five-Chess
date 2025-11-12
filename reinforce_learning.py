#!/usr/bin/env python

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from net import ConvNet
import os
import time
import matplotlib.pyplot as plt


BOARD_SIZE = 15
GAMMA = 0.99
LAMBDA = 0.95
CLIP_EPS = 0.2
VALUE_COEF = 0.5
ENTROPY_COEF = 0.01
PPO_EPOCHS = 4
BATCH_SIZE = 256

# Optimizer/Scheduler hyperparameters
INIT_LR = 3e-4
LR_MIN = 3e-5
LR_T_MAX = 2000  # steps for one cosine cycle (smaller for visible changes)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_model = ConvNet().to(device)
_optimizer = torch.optim.Adam(_model.parameters(), lr=INIT_LR)
try:
    _scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        _optimizer, T_max=LR_T_MAX, eta_min=LR_MIN
    )
except Exception as _e:
    print(f"Create scheduler failed: {_e}")
    _scheduler = None
print(f"Using device: {device}")
try:
    print(f"Torch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA runtime: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
except Exception as _e:
    print(f"Device info error: {_e}")

# Checkpoints directory
_ckpt_dir = os.path.join(os.path.dirname(__file__), "models")
_log_dir = os.path.join(os.path.dirname(__file__), "logs")
try:
    os.makedirs(_ckpt_dir, exist_ok=True)
    print(f"Model checkpoint dir: {_ckpt_dir}")
except Exception as _e:
    print(f"Create model dir failed: {_e}")
try:
    os.makedirs(_log_dir, exist_ok=True)
    print(f"Logs dir: {_log_dir}")
except Exception as _e:
    print(f"Create logs dir failed: {_e}")

# PPO storage buffer
_buffer = {
    'states': [],
    'actions': [],
    'log_probs': [],
    'values': [],
    'rewards': [],
    'dones': []
}

# Global training step and save tracking
_opt_step = 0
_has_saved_once = False

# Metrics
_game_count = 0
_black_win_count = 0
_win_rate_history = []
_game_index_history = []
_lr_steps = []
_lr_values = []

def _list_to_tensor_x(batch_x):
    # Input comes as list with shape (N, 15, 15, 2) or a single (15, 15, 2)
    arr = np.asarray(batch_x, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[None, ...]
    # Convert NHWC -> NCHW
    arr = np.transpose(arr, (0, 3, 1, 2))
    return torch.from_numpy(arr)


def reset_buffer():
    for key in _buffer:
        _buffer[key].clear()


def select_action(board, legal_moves, deterministic=False):
    """Select an action using the PPO policy. legal_moves as list of (x,y)."""
    x_tensor = _list_to_tensor_x(board).to(device)
    _model.eval()
    with torch.no_grad():
        logits, value = _model(x_tensor)
    logits = logits[0].cpu()
    value = value[0].item() if isinstance(value, torch.Tensor) else float(value)

    mask = torch.full((BOARD_SIZE * BOARD_SIZE,), float('-inf'))
    legal_indices = []
    for mv in legal_moves:
        idx = mv[0] * BOARD_SIZE + mv[1]
        legal_indices.append(idx)
    if len(legal_indices) == 0:
        # no moves available; return pass
        return 0, 0.0, value, []
    mask[legal_indices] = 0.0
    masked_logits = logits + mask
    probs = torch.softmax(masked_logits, dim=0)

    if deterministic:
        action = torch.argmax(probs).item()
        log_prob = torch.log(probs[action] + 1e-10).item()
    else:
        dist = Categorical(probs=probs)
        action = dist.sample().item()
        log_prob = dist.log_prob(torch.tensor(action)).item()

    return action, log_prob, value, probs.numpy()


def record_episode(transitions, winner_color):
    """Store one self-play episode into PPO buffer."""
    if len(transitions) == 0:
        return
    for idx, trans in enumerate(transitions):
        if winner_color == 0:
            reward = 0.0
        elif trans['color'] == winner_color:
            reward = 1.0
        else:
            reward = -1.0
        done = idx == len(transitions) - 1
        _buffer['states'].append(trans['state'])
        _buffer['actions'].append(trans['action'])
        _buffer['log_probs'].append(trans['log_prob'])
        _buffer['values'].append(trans['value'])
        _buffer['rewards'].append(reward)
        _buffer['dones'].append(done)


def _compute_advantages(values, rewards, dones):
    values = torch.tensor(values, dtype=torch.float32)
    rewards = torch.tensor(rewards, dtype=torch.float32)
    dones = torch.tensor(dones, dtype=torch.float32)
    advantages = torch.zeros_like(rewards)
    returns = torch.zeros_like(rewards)
    gae = 0.0
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_value = 0.0
            next_non_terminal = 0.0
        else:
            next_value = values[t + 1].item()
            next_non_terminal = 1.0 - dones[t].item()
        delta = rewards[t] + GAMMA * next_value * next_non_terminal - values[t]
        gae = delta + GAMMA * LAMBDA * next_non_terminal * gae
        advantages[t] = gae
        returns[t] = advantages[t] + values[t]
    return returns, advantages


def train():
    if len(_buffer['states']) == 0:
        return

    states_tensor = _list_to_tensor_x(_buffer['states']).to(device)
    actions_tensor = torch.tensor(_buffer['actions'], dtype=torch.long, device=device)
    old_log_probs = torch.tensor(_buffer['log_probs'], dtype=torch.float32, device=device)
    values_np = np.asarray(_buffer['values'], dtype=np.float32)
    rewards_np = np.asarray(_buffer['rewards'], dtype=np.float32)
    dones_np = np.asarray(_buffer['dones'], dtype=np.float32)

    returns_tensor, advantages_tensor = _compute_advantages(values_np, rewards_np, dones_np)
    returns_tensor = returns_tensor.to(device)
    advantages_tensor = advantages_tensor.to(device)
    advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)

    dataset_size = states_tensor.size(0)
    loss_info = []

    for epoch in range(PPO_EPOCHS):
        permutation = torch.randperm(dataset_size)
        for start in range(0, dataset_size, BATCH_SIZE):
            idx = permutation[start:start + BATCH_SIZE]
            batch_states = states_tensor[idx]
            batch_actions = actions_tensor[idx]
            batch_old_log_probs = old_log_probs[idx]
            batch_advantages = advantages_tensor[idx]
            batch_returns = returns_tensor[idx]

            _model.train()
            logits, values = _model(batch_states)
            dist = Categorical(logits=logits)
            new_log_probs = dist.log_prob(batch_actions)
            entropy = dist.entropy().mean()

            ratio = torch.exp(new_log_probs - batch_old_log_probs)
            surr1 = ratio * batch_advantages
            surr2 = torch.clamp(ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS) * batch_advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = F.mse_loss(values, batch_returns)
            loss = policy_loss + VALUE_COEF * value_loss - ENTROPY_COEF * entropy

            _optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(_model.parameters(), 1.0)
            _optimizer.step()
            current_lr = _optimizer.param_groups[0]['lr']
            if _scheduler is not None:
                try:
                    _scheduler.step()
                    # prefer scheduler-reported lr when available
                    last_lrs = _scheduler.get_last_lr()
                    if isinstance(last_lrs, (list, tuple)) and len(last_lrs) > 0:
                        current_lr = float(last_lrs[0])
                except Exception:
                    pass

            loss_info.append((policy_loss.item(), value_loss.item(), entropy.item()))

            global _opt_step
            _opt_step += 1
            try:
                _log_lr(_opt_step, current_lr)
            except Exception:
                pass
            if _opt_step % 100 == 0:
                path = save_model(step=_opt_step)
                print(f"saved checkpoint at opt_step={_opt_step}: {path}")

    if loss_info:
        p_loss = np.mean([x[0] for x in loss_info])
        v_loss = np.mean([x[1] for x in loss_info])
        entropy_avg = np.mean([x[2] for x in loss_info])
        print(f"PPO update -> policy_loss: {p_loss:.4f}, value_loss: {v_loss:.4f}, entropy: {entropy_avg:.4f}")

    global _has_saved_once
    if not _has_saved_once:
        path = save_model(step=_opt_step)
        print(f"saved first checkpoint after initial training: {path}")
        _has_saved_once = True

    reset_buffer()


def save_model(step: int | None = None, mark_latest: bool = True) -> str:
    """
    Save current model parameters.
    Returns saved path.
    """
    latest = os.path.join(_ckpt_dir, "latest.pt")
    tmp_name = f"tmp_{os.getpid()}_{int(time.time() * 1000)}.pt"
    tmp_path = os.path.join(_ckpt_dir, tmp_name)
    payload = {
        'model_state': _model.state_dict(),
        'meta': {
            'step': int(step) if step is not None else None,
            'timestamp': time.time()
        }
    }
    torch.save(payload, tmp_path)
    os.replace(tmp_path, latest)

    # remove any other checkpoint files to keep only the latest
    for fname in os.listdir(_ckpt_dir):
        fpath = os.path.join(_ckpt_dir, fname)
        if not fname.endswith('.pt'):
            continue
        if os.path.abspath(fpath) == os.path.abspath(latest):
            continue
        try:
            os.remove(fpath)
        except OSError:
            pass

    print(f"Model saved to: {latest}")
    return latest


def load_model(path: str) -> None:
    """Load model parameters from a checkpoint path into the global model."""
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    try:
        ckpt = torch.load(path, map_location=device)
    except Exception as exc:
        raise RuntimeError(f"Failed to read checkpoint '{path}': {exc}") from exc
    state_dict = ckpt.get('model_state') if isinstance(ckpt, dict) else None
    if state_dict is None:
        raise RuntimeError(f"Checkpoint '{path}' missing 'model_state' key")
    try:
        _model.load_state_dict(state_dict)
    except Exception as exc:
        raise RuntimeError(f"Checkpoint '{path}' incompatible with current model: {exc}") from exc
    _model.to(device)
    _model.eval()


def load_latest_model() -> bool:
    """Load the latest checkpoint if available. Returns True on success."""
    latest = os.path.join(_ckpt_dir, "latest.pt")
    if not os.path.isfile(latest):
        return False
    try:
        load_model(latest)
    except Exception as exc:
        print(f"Warning: failed to load latest checkpoint '{latest}': {exc}")
        return False
    return True


def _write_csv(path, header, rows):
    exists = os.path.isfile(path)
    with open(path, 'a', encoding='utf-8') as f:
        if not exists:
            f.write(','.join(header) + "\n")
        for r in rows:
            def _fmt(x):
                if isinstance(x, float):
                    return f"{x:.8f}"
                return str(x)
            f.write(','.join(_fmt(x) for x in r) + "\n")


def _plot_curve(xs, ys, title, xlab, ylab, out_path):
    if len(xs) == 0:
        return
    plt.figure(figsize=(6, 4))
    plt.plot(xs, ys)
    plt.title(title)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def log_game_result(winner_color: int) -> None:
    global _game_count, _black_win_count
    _game_count += 1
    if winner_color == 1:
        _black_win_count += 1
    win_rate = _black_win_count / max(1, _game_count)
    _game_index_history.append(_game_count)
    _win_rate_history.append(win_rate)

    try:
        _write_csv(
            os.path.join(_log_dir, 'win_rate.csv'),
            ['game', 'black_wins', 'win_rate'],
            [( _game_count, _black_win_count, win_rate )]
        )
        _plot_curve(_game_index_history, _win_rate_history,
                    'Self-play Win Rate (Black)', 'Game', 'Win Rate',
                    os.path.join(_log_dir, 'win_rate.png'))
    except Exception:
        pass


def _log_lr(step: int, lr_value: float) -> None:
    _lr_steps.append(step)
    _lr_values.append(lr_value)
    if step % 10 == 0:
        try:
            _write_csv(
                os.path.join(_log_dir, 'learning_rate.csv'),
                ['step', 'lr'],
                [(step, lr_value)]
            )
            _plot_curve(_lr_steps, _lr_values,
                        'Learning Rate', 'Step', 'LR',
                        os.path.join(_log_dir, 'learning_rate.png'))
        except Exception:
            pass




