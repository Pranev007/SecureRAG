# SecureRAG - common tasks.
#
# Everything here works with no external services: the backend falls back to
# SQLite and the offline providers, so `make test` and `make eval` run on a
# clean checkout with nothing installed but Python and Node.

.DEFAULT_GOAL := help
PY := python
BACKEND := backend
FRONTEND := frontend

.PHONY: help
help: ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

.PHONY: install
install: ## Install backend and frontend dependencies
	cd $(BACKEND) && $(PY) -m pip install -r requirements-dev.txt
	cd $(FRONTEND) && npm install

.PHONY: migrate
migrate: ## Apply database migrations
	cd $(BACKEND) && alembic upgrade head

.PHONY: dev
dev: ## Run the backend with reload (SQLite, no infrastructure needed)
	cd $(BACKEND) && DATABASE_URL=sqlite:///./securerag_dev.db \
		uvicorn app.main:app --reload --port 8000

.PHONY: web
web: ## Run the frontend dev server
	cd $(FRONTEND) && npm run dev

.PHONY: test
test: ## Run the full backend test suite
	cd $(BACKEND) && pytest -q

.PHONY: test-security
test-security: ## Run only the adversarial security tests
	cd $(BACKEND) && pytest tests/security -v

.PHONY: lint
lint: ## Lint and format-check both sides
	cd $(BACKEND) && ruff check . && ruff format --check .
	cd $(FRONTEND) && npm run lint && npm run typecheck

.PHONY: format
format: ## Auto-format the backend
	cd $(BACKEND) && ruff check . --fix && ruff format .

.PHONY: eval
eval: ## Run the evaluation suite and write a report
	cd $(BACKEND) && $(PY) -m app.evaluation.run

.PHONY: demo
demo: ## Seed a demo workspace against a running API
	$(PY) scripts/seed_demo.py

.PHONY: up
up: ## Start the whole stack in Docker
	docker compose up --build

.PHONY: down
down: ## Stop the stack
	docker compose down

.PHONY: clean
clean: ## Remove caches, build output and local databases
	find . -type d -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null || true
	rm -rf $(BACKEND)/.pytest_cache $(BACKEND)/.ruff_cache $(BACKEND)/htmlcov
	rm -rf $(BACKEND)/*.db $(BACKEND)/storage $(FRONTEND)/dist
