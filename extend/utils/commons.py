import subprocess
from typing import Optional

import logging

logger = logging.getLogger(__name__)


def execute_bash_command(command: str) -> Optional[str]:
    command_result = subprocess.run(command, shell=True, capture_output=True)
    try:
        command_result.check_returncode()
        return command_result.stdout.decode("utf-8")
    except subprocess.CalledProcessError:
        logger.warning(f"failed executing command: {command}")
        logger.warning(f"return code was: {command_result.returncode}")
        logger.warning(f'stdout was: {command_result.stdout.decode("utf-8")}')
        logger.warning(f'stderr code was: {command_result.stderr.decode("utf-8")}')
        return None
