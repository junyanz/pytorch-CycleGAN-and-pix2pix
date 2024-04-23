SHELL = bash
.DEFAULT_GOAL := help

export COMPOSE_DOCKER_CLI_BUILD=1
export DOCKER_BUILDKIT=1

# Determine if Docker Compose v2 is available
ifneq (, $(shell docker --help | grep compose))
DC := docker compose
else
DC := docker-compose
endif

# Find a good Docker PS tool
ifneq (, $(shell which lazydocker))
DOCKER_PS := lazydocker
else
ifneq (, $(shell which ctop))
DOCKER_PS := ctop
else
DOCKER_PS := watch ${DC} ps
endif
endif


.PHONY: bootstrap
bootstrap: ## Bootstrap local repository checkout
	@echo Set git commit message template…
	git config commit.template .gitmessage

	@echo Installing Python dependencies into Poetry managed virtualenv
ifeq (, $(shell which poetry))
	@echo "No \`poetry\` in \$$PATH, please install poetry https://python-poetry.org"
else
	. ./ca_token.sh && poetry install --no-root
endif

	poetry run pre-commit install


.PHONY: docker-build
docker-build: ## Build docker image for local development (alias: build)
	. ./ca_token.sh && ${DC} build


.PHONY: build
build: docker-build


.PHONY: docker-up
docker-up: ## Start composed docker services for local development (alias: up)
	${DC} up --remove-orphans --detach
	${DOCKER_PS}


.PHONY: up
up: docker-up


.PHONY: docker-down
docker-down: ## Stop composed docker services for local development (alias: down)
	${DC} down --volumes --remove-orphans


.PHONY: down
down: docker-down
