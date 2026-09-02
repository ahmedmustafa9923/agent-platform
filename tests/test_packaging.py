from pathlib import Path

import agent_rag

def test_package_imports_from_src():
    assert Path(agent_rag.__file__).parent.name == "agent_rag"
def test_package_directory_is_named_agent_rag():

    assert Path(agent_rag.__file__).parents[1].name == "src"
