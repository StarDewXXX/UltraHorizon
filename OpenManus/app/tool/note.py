from typing import Dict, List
from app.tool.base import BaseTool

class NoteTool(BaseTool):
    """A tool for writing and retrieving notes."""

    name: str = "note_tool"
    description: str = "A tool to write and view notes. You can add a note or list all existing notes. You may need to record your plans, ideas, or important information. And **it's recommended to check notes frequently** to keep track of your thoughts and tasks."
    parameters: dict = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["write_note", "check_notes"],
                "description": "The action to perform: write_note to add a note, check_notes to view all notes.",
            },
            "note": {
                "type": "string",
                "description": "The note content to write. Required if action is write_note.",
            },
        },
        "required": ["action"],
    }

    # A simple in-memory storage for notes
    notes: List[str] = []

    async def execute(
        self,
        action: str,
        note: str = None,
    ) -> Dict:
        """
        Executes the note tool.

        Args:
            action (str): Either "write_note" or "check_notes".
            note (str, optional): The note content to write. Required if action is "write_note".

        Returns:
            Dict: Response dict with success status and message or notes.
        """
        if action == "write_note":
            if not note:
                return {
                    "success": False,
                    "message": "Note content is required when action is write_note.",
                }
            self.notes.append(note)
            return {
                "success": True,
                "message": f"Note added: {note}",
            }

        elif action == "check_notes":
            return {
                "success": True,
                "notes": self.notes,
            }

        else:
            return {
                "success": False,
                "message": f"Unsupported action: {action}",
            }
