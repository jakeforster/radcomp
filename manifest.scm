;; packages for Radcomp development environment
;; usage: guix shell [-C] manifest.scm
;; for plotting, use: export MPLBACKEND=webagg

(specifications->manifest '("python" "python-numpy" "python-scipy" "python-matplotlib" "python-tornado" "python-pytest"))
