import sys

import gamma_v3_custom_env

sys.modules["custom_env"] = gamma_v3_custom_env

from scripts.evaluate_checkpoint import main


if __name__ == "__main__":
    main()

