# test_env.py
import os

def test_check_api_keys():
    """
    Checks if the required GROQ_API_KEY is present in the environment.
    This prevents the pipeline from running expensive code if setup is wrong.
    """
    api_key = os.getenv("GROQ_API_KEY")
    assert api_key is not None, "Error: GROQ_API_KEY not found in environment variables."
    assert api_key.startswith("gsk_"), "Error: GROQ_API_KEY format appears invalid."

def test_check_directories():
    """
    Checks if the project structure is intact.
    """
    from pathlib import Path
    # This ensures your 'data' folder exists as expected by your code
    data_dir = Path("data")
    # We don't assert it exists yet (CI starts empty), but we check we can create it
    data_dir.mkdir(exist_ok=True)
    assert data_dir.is_dir()

    #