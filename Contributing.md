# Contributing to Voice AI Agent 🎙️📞

Thank you for your interest in contributing to **Voice AI Agent**!  
We welcome contributions that improve features, reliability, documentation, and developer experience.


## 🚀 Getting Started

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/psyuktha/voice_ai.git
cd voice_ai
```
### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```
### 3️⃣ Run FastAPI Backend (Vapi Agent)
```bash
uvicorn main:app --reload
```
⚠️ Ensure webhook payloads match the schema expected by `vapi_workflow.json`.

## 🚀 How to Contribute

We welcome contributions from the community! Here's how you can get involved:

### 🐞 Reporting Issues

If you encounter bugs or have feature requests, please open an issue with:
- A clear description
- Steps to reproduce (if applicable)
- Expected vs actual behavior

### 💡 Feature Requests

We’d love to hear your ideas!  
Submit a feature request by opening an issue and explaining:
- The problem it solves
- Why it’s useful
- Any implementation ideas (optional)

### 🧑‍💻 Code Contributions

Follow these steps to contribute code:

#### 1️⃣ Fork the Repository
Click the **Fork** button at the top-right of the repository page.

#### 2️⃣ Create a New Branch
```bash
git checkout -b feature/your-feature-name
```
#### 3️⃣ Make Changes
Write clean, well-documented code
Follow existing project structure
Ensure all tests pass (if applicable)

#### 4️⃣ Commit Changes
```bash
git commit -m "Add a meaningful commit message"
```
### 5️⃣ Push to Your Branch
```bash
git push origin feature/your-feature-name
```
### 6️⃣ Submit a Pull Request
Go to the original repository and click New Pull Request.
Make sure to provide:
A clear description of the changes
Screenshots or examples (if applicable)

## 🧑‍💻 Code Style Guidelines

To keep the codebase clean and maintainable, please follow these guidelines:

- Follow **PEP8** style guidelines for Python code
- Use meaningful and descriptive variable, function, and class names
- Keep functions small and focused on a single responsibility
- Ensure code is modular and easy to extend
- Add comments for complex logic, especially in:
  - Intent classification
  - Entity extraction
  - Conversation flow handling
  - Post-call summarization
- Avoid hardcoding values; use configuration files or environment variables where appropriate


## 📚 Best Practices

- Write clear and meaningful commit messages (e.g., `feat: add Gemini intent parser`)
- Ensure changes align with the existing project architecture
- Maintain consistency between:
  - Vapi workflow logic
  - FastAPI webhook handling
  - Gemini agent modules
- Avoid introducing unnecessary dependencies
- Update documentation (`README.md`, sample JSONs) if behavior changes

Following these guidelines helps ensure high-quality contributions and smooth collaboration 🚀

Thank you for contributing and helping improve this project! 🙌
