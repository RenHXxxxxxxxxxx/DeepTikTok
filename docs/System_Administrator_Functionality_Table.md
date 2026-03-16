# System Administrator Functionality Table

| Module | Function Name | Description | Implementation Reality |
|---|---|---|---|
| Identity & Access | Admin Login | Administrator logs into the backend to manage the system. | Uses Django's built-in `/admin/login/` page and the `User` model verification. |
| Identity & Access | Register User | Administrator manually creates a new staff or user account. | Uses the "Add User" button in Django Admin, setting the `is_staff` check box. |
| User Management | Disable Creator Account | Blocks a specific TikTok creator from accessing the system. | Unchecking the "Active" boolean field in the Django Admin `User` model. |
| User Management | Reset Password | Changes the password for a user who forgot their credentials. | Utilizing the built-in password reset form within Django Admin. |
| Data Management | View Scraped Data List | Browses the raw video and comment data collected from TikTok. | Standard list display of the scraped `DouyinData` or `Video` Django models. |
| Data Management | Delete Invalid Records | Deletes scraped database rows that have missing or broken fields. | Selecting rows and clicking the default "Delete selected" dropdown action. |
| Task Scheduling | Start Scraping Task | Manually triggers the data collection process for a target creator. | A custom Django Admin view button that invokes the DrissionPage scraping script. |
| ML Pipeline | Trigger XGBoost Training | Starts training the machine learning model on the latest dataset. | A custom Django view that triggers the local `train_master_arena.py` script. |
| ML Pipeline | Switch Active Model | Selects which trained XGBoost model file to use for predictions. | Updating a file path string in a custom global configuration Django model. |
| Storage & System | Clear Temp Video Files | Deletes raw `.mp4` video downloads to free up hard drive space. | A custom admin button executing `os.remove()` inside the `/media/` folder. |
| Storage & System | View API Error Logs | Checks logs for DeepSeek token limits or network request errors. | A simple Django view that reads and displays the `errors.log` text file. |
