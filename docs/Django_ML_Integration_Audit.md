# Django & ML Integration Audit Report

## 1. API View Target (`views.py`)
- **Finding:** The view function `retrain_model_api` handling the `/api/retrain_model/` endpoint contains a hardcoded reference to the legacy training script. In `views.py` (around line 2158), the subprocess execution logic specifically targets `train_model_arena.py`.
- **Action Required:** This hardcoded path must be updated by the developer to target the newly refactored unified engine, `train_master_arena.py`.

## 2. The `AppConfig` Threading Trap (`apps.py`)
- **Mechanics of the Trap:** When an independent ML training script needs to access the database, it calls `django.setup()`. This boot process forces Django to initialize all registered applications, ultimately triggering the `ready()` method in `apps.py`. Because `ready()` is currently set up to call `start_ai_worker()`, it inadvertently spawns the background `AIWorker` daemon threads right inside the ML subprocess. This causes the ML script to hang, crash, or experience resource collisions as the daemon continuously polls the database in a purely analytical environment.
- **Theoretical Solution:** To block the `AIWorker` from spawning during ML training, implement an execution context bypass. The ML script should set a custom environment variable, such as `os.environ['IS_ML_TRAINING'] = 'True'`, prior to calling `django.setup()`. The `ready()` method in `apps.py` can then be modified to include a condition like `if os.environ.get('IS_ML_TRAINING') != 'True':` before executing `start_ai_worker()`, effectively disabling the daemon when the app is booted purely for ORM access.
