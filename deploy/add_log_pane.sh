#!/usr/bin/env bash
# Add a live log pane under the running bot in its tmux session, without touching the bot.
#
#   bash deploy/add_log_pane.sh [session]        # run from a normal SSH shell, NOT inside tmux
#
# What it does (only tmux layout commands; the bot's process is never signalled or restarted):
#   1. finds the pane running the bot (a python process) in the session
#   2. splits that window and runs watch_logs.py below it on the bot's logs/ directory, in a loop
#      so the viewer comes back if it is ever stopped (the bot's checkout is not modified)
#   3. nothing else: the pane shares the bot's window, so `tmux attach` shows both (no tmux hooks;
#      one crashed the tmux server on the contest box)
# Safe to run twice: it does nothing if the log pane already exists.
set -euo pipefail

SESSION="${1:-$(tmux list-sessions -F '#S' | head -n 1)}"
TITLE="trend-guard-logs"

TITLES="$(tmux list-panes -s -t "$SESSION" -F '#{pane_title}')"
if [[ $'\n'"$TITLES"$'\n' == *$'\n'"$TITLE"$'\n'* ]]; then  # exact line match, no pipe
  echo "log pane already exists in session '$SESSION'"
  exit 0
fi

# the bot's pane: the first pane in the session whose foreground command is python
BOT_PANE="$(tmux list-panes -s -t "$SESSION" -F '#{pane_id} #{pane_current_command}' \
  | awk '$2 ~ /^python/ {print $1; exit}')"
if [ -z "$BOT_PANE" ]; then
  echo "no pane running python found in session '$SESSION'; panes are:" >&2
  tmux list-panes -s -t "$SESSION" -F '  #{pane_id} #{window_index}.#{pane_index} #{pane_current_command}' >&2
  exit 1
fi
BOT_DIR="$(tmux display-message -p -t "$BOT_PANE" '#{pane_current_path}')"
# the viewer: next to this script, in a repo layout beside it, or in the bot's own checkout
HERE="$(cd "$(dirname "$0")" && pwd)"
WATCH=""
for f in "$HERE/watch_logs.py" "$HERE/../scripts/watch_logs.py" "$BOT_DIR/scripts/watch_logs.py"; do
  if [ -f "$f" ]; then WATCH="$(cd "$(dirname "$f")" && pwd)/watch_logs.py"; break; fi
done
if [ -z "$WATCH" ]; then
  echo "watch_logs.py not found next to this script or in $BOT_DIR/scripts" >&2
  exit 1
fi

VIEWER="while true; do python3 '$WATCH' --logs '$BOT_DIR/logs'; echo 'viewer stopped; restarting in 3 s'; sleep 3; done"
# -d keeps focus where it is while we set up; -l 45% gives the logs a bit under half the window
LOG_PANE="$(tmux split-window -d -v -l 45% -t "$BOT_PANE" -c "$BOT_DIR" -P -F '#{pane_id}' "$VIEWER")"
tmux select-pane -t "$BOT_PANE" -T "trend-guard-bot"
tmux select-pane -t "$LOG_PANE" -T "$TITLE"  # last, so the log pane is the active one

# No tmux hooks: a client-attached hook (select-window / select-pane) crashed the tmux server on the
# contest EC2 (Amazon Linux 2023, tmux 3.2a) and took the bot with it. The log pane shares the bot's
# window, so a plain `tmux attach` already shows it.
WINDOW="$(tmux display-message -p -t "$BOT_PANE" '#{session_name}:#{window_index}')"

echo "bot pane $BOT_PANE ($BOT_DIR), log pane $LOG_PANE added to $WINDOW"
echo "attach with: tmux attach -t $SESSION"
