# convcontract

An AI assistant makes more errors when a user gives a task in many small messages. Laban et al. show this problem in *LLMs Get Lost in Multi-Turn Conversation* (ICLR 2026). This project tests one solution to this problem: the conversation contract. A helper model (the extractor) keeps a list of all the items that the user asks for. The main model (the executor) uses only this list to make its answer. The executor does not see the chat history, and it does not see its own previous answers.

The extractor and the executor are usual LLM calls through OpenRouter. Model training is not necessary. On Microsoft's `lost_in_conversation` harness (gpt-4.1-nano, 6+ shards), the contract increases BFCL tool-use accuracy from 40% to 73%. The coding result (HumanEval) is not conclusive, and math (GSM8K) shows no gain. For the results, refer to [docs/findings.md](docs/findings.md). For the method, refer to [docs/method.md](docs/method.md).
