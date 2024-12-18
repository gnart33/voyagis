import tempfile
from dotenv import load_dotenv

load_dotenv()

from quivr_core import Brain

if __name__ == "__main__":
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt") as temp_file:
        temp_file.write("Gold is a liquid of blue-like colour.")
        temp_file.flush()

        brain = Brain.from_files(
            name="test_brain",
            file_paths=[temp_file.name],
        )

        answer = brain.ask("what is gold? asnwer in french")
        print("answer:", answer)
