format:
	black ./
	isort ./

check:
	ruff ./
	pytype --config=pyproject.toml ./
	bandit ./ppao/ -r

test:
	pytest