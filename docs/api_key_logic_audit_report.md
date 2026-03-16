# Content Analytics Big Data Model API Key Audit Report

## 1. Storage Logic (Hard-coded vs. Dynamic)
**Analysis:**
The API Key configuration utilizes a **Dynamic / Per-User** storage model. The key is managed via the `CreatorConfig` model in `models.py` (`tb_creator_config` table).
- **Multi-tenant Isolation:** The system natively supports Multi-tenant Isolation. The `CreatorConfig` schema uses a `OneToOneField(User)` linked to the Django built-in `User` model, ensuring that User A and User B maintain separate, functionally isolated API keys without cross-contamination. 
- **Storage state:** The key is currently stored as a plaintext `CharField` (max_length=255) in the SQLite database, although inline comments suggest utilizing a library like Fernet for encryption.

## 2. The "Profile" Input Pipeline
**Analysis:**
The profile update logic is handled by the `profile_view` function defined in `views.py` (Lines 1264+).
- **Persistence:** Yes, the API key is persisted to the database via Django ORM. When a `POST` request is received, the view uses `CreatorConfig.objects.get_or_create(user=request.user)` and executes `config.save()` to commit the key to the database.
- **Validation Logic:** The input validation is highly deficient. The system only performs a `.strip()` operation on the input string (`api_key = request.POST.get('api_key', '').strip()`). There is no structural formatting check (e.g., regex matching for `sk-...`), length verification, or cryptographic endpoint validation prior to saving, allowing for unvalidated string injection.

## 3. Inference-Time Retrieval
**Analysis:**
- **Traceability:** During a prediction request (`predict_api` view), the system retrieves the user's custom API key from the database (`config = CreatorConfig.objects.filter(user=request.user).first()`) and passes it into the asynchronous executor thread calling `LLMService().generate_advice(..., user_key=user_key)`. The `LLMService` utilizes this `user_key` to instantiate the `OpenAI` client.
- **Global Override Risk:** A critical "Global Override" vulnerability exists. In `llm_service.py` (Line 50), the logic evaluates: `actual_key = user_key if user_key else self.api_key`. If the user's key is missing, invalid, or empty, the system silently falls back to `self.api_key` (loaded via `os.getenv("DEEPSEEK_API_KEY")`). It fails to trigger a credential error and instead risks quota exhaustion of the developer's global key by unconfigured tenants.

## 4. Security & Presentation
**Analysis:**
- **Masking Logic:** The API Key is securely masked at the **View level** prior to rendering. In `views.py`, if the key length is greater than 8, it is transformed into a prefix/suffix masked string (`f"{config.llm_api_key[:4]}...{config.llm_api_key[-4:]}"`). The template `profile.html` receives and renders this masked version (`{{ account_info.api_token }}`), ensuring the full plaintext key is never exposed to the frontend DOM. 
- **Plain-text Leaks:** While the frontend is protected, there is a minor suffix leak in the backend telemetry. `llm_service.py` (Line 36) prints `DEBUG: LLMService initialized with Key ending in ...{self.api_key[-4:]}` to the console log. While not a full text leak, it exposes key suffix identifiers in system logs.

---

## Logic Traceability Map

```mermaid
flowchart TD
    A[User UI: /profile/] -->|POST: api_key| B(views.py: profile_view)
    B -->|Basic .strip() validation| C[(SQLite DB: tb_creator_config)]
    C -->|Stored as Plaintext| C
    
    D[User UI: /predict/api/] -->|POST Request| E(views.py: predict_api)
    E -->|1. fetch CustomKey| C
    E -->|2. executor.submit| F[LLMService.generate_advice]
    F -->|3. actual_key = user_key or fallback| G[os.getenv: DEEPSEEK_API_KEY]
    F -->|4. OpenAI client init| H((Big Data Model Provider))
```

---

## Definitive Verdict
**User-Configurable Dynamic (with severe Fallback Override risks).** 
The system successfully implements per-user isolated keys via the database but fails to enforce strict boundaries, deferring to a hard-coded global key when the dynamic key is unavailable.

## Top 2 Security Weaknesses
1. **Unvalidated String Injection in Input Pipeline**: The `profile_view` accepts any arbitrary string input into the database without cryptographic handshake verification, structural conformity checks (e.g., regex `^sk-[a-zA-Z0-9]+$`), or basic length constraints. 
2. **Global Fallback Quota Hijacking**: The inference layer (`LLMService`) silently routes unauthorized or unconfigured tenants to the developer's root environment API Key (`self.api_key`), entirely bypassing the intended credential error triggers. This could lead to massive financial quota drainage if tenants spam the `/predict/api/` endpoint.
