
import os
from pathlib import Path

FILE_PATH = str(Path(__file__).parent.parent / "src" / "core" / "event_processor.py")

INIT_CODE = """
    def __init__(self, telegram_client):
        self.client = telegram_client
        self.personality_db = PersonalityDatabase()
        self.analyzer = MessageAnalyzer()
        self.contacts = ContactsManager()
        self.command_parser = CommandParser()
        self.llm = LLMInterface()
        self.behavior = HumanBehaviorSimulator()

        # Access Control
        self.permissions = PermissionsManager()

        # Initialize tools
        self.tools_manager = ToolsManager()
        self._register_tools()

        # Locks to prevent parallel responses
        self.chat_locks = defaultdict(Lock)
        self.processing_status = {}

        logger.info("Event processor initialized with tools support")

"""

IMPORT_LINE = "from src.core.permissions import PermissionsManager"

def fix_file():
    with open(FILE_PATH, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # 1. Add import if missing
    if not any("PermissionsManager" in line for line in lines[:30]):
        # Find last import
        insert_idx = 0
        for i, line in enumerate(lines[:30]):
            if line.startswith("from") or line.startswith("import"):
                insert_idx = i
        lines.insert(insert_idx + 1, IMPORT_LINE + "\n")
        print("Added import.")

    # 2. Find start and end of garbage block
    start_idx = -1
    end_idx = -1

    for i, line in enumerate(lines):
        if "COMPLEXITY_INSTRUCTIONS = {" in line:
            # Skip forward to closing brace
            for j in range(i, len(lines)):
                if "}" in lines[j]:
                    start_idx = j + 1
                    break
            break

    for i, line in enumerate(lines):
        if "def _register_tools(self):" in line:
            end_idx = i
            break

    if start_idx != -1 and end_idx != -1:
        print(f"Replacing block from line {start_idx} to {end_idx}")
        # Keep lines before start
        new_lines = lines[:start_idx]
        # Insert init code
        new_lines.append(INIT_CODE)
        # Keep lines after end
        new_lines.extend(lines[end_idx:])

        with open(FILE_PATH, 'w', encoding='utf-8') as f:
            f.writelines(new_lines)
        print("File repaired successfully.")
    else:
        print("Could not find block boundaries.")

if __name__ == "__main__":
    fix_file()
