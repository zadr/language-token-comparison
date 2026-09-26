PY := .venv/bin/python
KIMI_SHA256 := b6c497a7469b33ced9c38afb1ad6e47f03f5e5dc05f15930799210ec050c5103

.PHONY: all local anthropic page test shot

all: local anthropic page

VENV := .venv/.installed

$(VENV): requirements.txt
	uv venv -q .venv
	uv pip install -q --python $(PY) -r requirements.txt
	touch $@

data/RosettaCodeData/.git:
	git submodule update --init --depth 1 data/RosettaCodeData

# Kimi K2 through K3 ship this same file.
tok/kimi/tiktoken.model:
	mkdir -p $(@D)
	curl -sSfL -o $@ https://huggingface.co/moonshotai/Kimi-K3/resolve/main/tiktoken.model
	echo "$(KIMI_SHA256)  $@" | shasum -a 256 -c

build/files.jsonl: $(VENV) data/RosettaCodeData/.git
	$(PY) scripts/inventory.py

local: build/files.jsonl tok/kimi/tiktoken.model
	$(PY) scripts/tokenize_local.py

# Resumable; rerun after interruptions.
anthropic: build/files.jsonl
	$(PY) scripts/tokenize_anthropic.py

page:
	$(PY) scripts/build_data.py

test:
	node web/model.test.mjs

shot:
	sh scripts/shot.sh light
	sh scripts/shot.sh dark
