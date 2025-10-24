git fetch --all --prune
git reset --hard origin/main
git clean -fd

git add -A
git commit -m "change skip data sc and pro"
git push origin main