#!/usr/bin/env python
"""Basic test of CPU model."""
import logging
from agent.cpu_local_model import CPULocalModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("Step 1: Loading model...")
model = CPULocalModel()
logger.info("Model loaded.")

logger.info("Step 2: Testing inference...")
messages = [
    {"role": "system", "content": "You are helpful."},
    {"role": "user", "content": "What is 2+3?"},
]
outputs = model(messages, temperature=0.0, max_tokens=50)
logger.info(f"Generated: {outputs[0][:100]}")
logger.info(f"Last seq logprob: {model.last_seq_logprob}")
logger.info("Done!")
