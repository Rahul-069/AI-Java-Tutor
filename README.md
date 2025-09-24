# 🤖 AI Java Tutor

An interactive **AI-powered Java Tutor** built with **Gradio** and the **DeepSeek Coder model**.  
It provides a chat-based learning experience, progress dashboards, and interactive quizzes to help students learn Java effectively.

---

## ✨ Features
- 💬 **Chat Interface** – Ask any Java-related question and get clear, beginner-friendly explanations with examples.  
- 📊 **Learning Dashboard** – Track number of questions asked, code examples provided, and topics covered.  
- 🧠 **Quiz Center** – Test your Java knowledge with multiple-choice quizzes (Basics, OOP, Advanced, or Overall).  
- 🎓 **Java Quick Reference** – Built-in guide with syntax and examples for key Java concepts.  
- 🎨 **Modern UI** – Glassmorphism + responsive design using custom Gradio theming.  

---

## 📂 Project Structure
```
AI-Java-Tutor/
├── javaaitutor.py      # Main application code (Gradio app + AI Tutor logic)
├── requirements.txt    # List of dependencies
└── README.md           # Project documentation
```


---

## 🚀 Getting Started

### 📌 Requirements
- Python 3.8+
- Gradio  
- Torch with CUDA (recommended for GPU acceleration)  
- Transformers  
- Accelerate  

---

### 1️⃣ Clone the repository
```bash
git clone https://github.com/<Rahul-069>/AI-Java-Tutor.git
cd AI-Java-Tutor
```

### 2️⃣ **Install dependencies**
Run this in your Python environment:
```
pip install -r requirements.txt
```

### 3️⃣ **Run the app**
```
python javaaitutor.py
```

##📌 **Note:**
If running on Google Colab, open the notebook version, set runtime to T4 GPU, and run all cells.
Gradio will provide a public link to access the app.

## 🙌 Conclusion
This project demonstrates how AI models can be integrated with Gradio to create an interactive Java tutoring system.  
