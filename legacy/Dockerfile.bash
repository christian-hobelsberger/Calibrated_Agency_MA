# Minimal Ubuntu sandbox for the AgentBench OS experiment.
# The bash MCP server runs inside this container for safe command execution.

FROM ubuntu:22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip \
    bash coreutils grep findutils sed gawk \
    && rm -rf /var/lib/apt/lists/*

# Create the agent home directory
RUN mkdir -p /tmp/agent_home && chmod 777 /tmp/agent_home

WORKDIR /tmp/agent_home

# Drop privileges for safety
USER nobody
