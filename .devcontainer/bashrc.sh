# ~/.bashrc - Portable and Container-Safe

# Exit if not interactive
case $- in
    *i*) ;;
    *) return;;
esac

# History and shell behavior
HISTCONTROL=ignoreboth
shopt -s histappend
HISTSIZE=1000
HISTFILESIZE=2000
shopt -s checkwinsize

# Prompt
if [ -n "$force_color_prompt" ]; then
    if [ -x /usr/bin/tput ] && tput setaf 1 &>/dev/null; then
        color_prompt=yes
    fi
fi

if [ "$color_prompt" = yes ]; then
    PS1='\u@\h:\w\$ '
else
    PS1='\u@\h:\w\$ '
fi

case "$TERM" in
    xterm*|rxvt*)
        PS1="\[\e]0;\u@\h: \w\a\]$PS1"
        ;;
esac

# Aliases and color support
if [ -x /usr/bin/dircolors ]; then
    test -r ~/.dircolors && eval "$(dircolors -b ~/.dircolors)" || eval "$(dircolors -b)"
    alias ls='ls --color=auto'
    alias grep='grep --color=auto'
    alias fgrep='fgrep --color=auto'
    alias egrep='egrep --color=auto'
fi

alias ll='ls -alF'
alias la='ls -A'
alias l='ls -CF'


# Enable programmable completion
if ! shopt -oq posix; then
    if [ -f /usr/share/bash-completion/bash_completion ]; then
        . /usr/share/bash-completion/bash_completion
    elif [ -f /etc/bash_completion ]; then
        . /etc/bash_completion
    fi
fi

# Container-safe Git branch prompt
parse_git_branch() {
    git branch 2>/dev/null | sed -n '/\* /s///p'
}
PS1="\u \[\e[32m\]\w \[\e[91m\]\$(parse_git_branch)\[\e[00m\]$ "

echo "🛠️ Running inside a Dev Container"
