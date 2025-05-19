#!/bin/bash

BASHRC="$HOME/.bashrc"

# Add the function and prompt only if not already present
if ! grep -q "parse_git_branch()" "$BASHRC"; then
    {
        echo ""
        echo "# Git branch in prompt"
        echo "parse_git_branch() {"
        echo "    git branch 2>/dev/null | sed -n '/\\* /s///p' | sed 's/.*/(\\0)/'"
        echo "}"
        echo 'PS1="\u \[\e[32m\]\w \[\e[91m\]\$(parse_git_branch)\[\e[00m\]\$ "'
    } >> "$BASHRC"
fi
