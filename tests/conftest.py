import os


if not os.getenv("GEMINI_API_KEY"):
    os.environ["GEMINI_API_KEY"] = "test-key"