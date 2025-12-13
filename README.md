# 🎓 Cambridge A-Level Math Solver

<div align="center">

**An AI-powered mathematics tutor that solves A-Level problems step-by-step using LangChain agents and Python execution**

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-0.2.0+-green.svg)](https://www.langchain.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Cloud-red.svg)](https://streamlit.io/)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o-orange.svg)](https://openai.com/)

[Live Demo](#) • [Documentation](#features) • [Report Bug](#)

</div>

---

## 📖 Overview

**Cambridge A-Level Math Solver** is an intelligent tutoring system that combines Large Language Models (LLMs) with symbolic computation to provide step-by-step solutions for A-Level mathematics problems. Unlike traditional calculators, this system uses an **AI agent** that reasons through problems, writes Python code, executes it, and explains the solution process—making it an ideal learning companion for students.

### 🎯 Key Innovation

This project demonstrates **production-grade AI agent development** by solving a critical challenge: preventing LLM hallucination. The agent is constrained to execute actual Python code (using SymPy, NumPy, Matplotlib) rather than guessing answers, ensuring mathematical accuracy and educational value.

---

## ✨ Features

### 🧮 Core Capabilities
- **Step-by-Step Solutions**: Breaks down complex problems into clear, educational steps
- **Symbolic Mathematics**: Handles calculus, algebra, integration, differentiation using SymPy
- **Visual Graphing**: Generates publication-quality plots using Matplotlib
- **Image Recognition**: Upload screenshots/photos of problems using GPT-4o Vision API
- **Code Transparency**: View the exact Python code executed for each solution

### 🎨 User Experience
- **Premium UI**: Modern, responsive Streamlit interface with chat history
- **Conversational Context**: Maintains conversation history for follow-up questions
- **Multi-Modal Input**: Text input or image upload
- **Real-Time Execution**: See Python code run in real-time with intermediate steps

### 🔒 Production Features
- **Secure API Key Management**: Environment variables and Streamlit secrets
- **Error Handling**: Robust error handling with user-friendly messages
- **Deployment Ready**: Configured for Streamlit Community Cloud

---

## 🏗️ Architecture

### System Design

```
┌─────────────────┐
│   Streamlit UI  │  ← User Interface (app.py)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  LangChain Agent │  ← AI Agent (agent.py)
│  (GPT-4o-mini)  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Python REPL    │  ← Code Execution
│  (SymPy/NumPy)  │
└─────────────────┘
```

### How It Works

1. **User Input**: Problem entered via text or image upload
2. **Image Processing** (if applicable): GPT-4o Vision transcribes the problem
3. **Agent Reasoning**: LangChain agent analyzes the problem and generates Python code
4. **Code Execution**: Python REPL executes code using SymPy/NumPy/Matplotlib
5. **Result Display**: Solution, steps, and graphs displayed in the UI

### Technical Highlights

- **Agent Framework**: Uses `create_python_agent` from LangChain Experimental for stable, cross-version compatibility
- **Prompt Engineering**: Strict prompts enforce code execution and prevent hallucination
- **Tool Integration**: PythonREPLTool provides sandboxed code execution
- **State Management**: Streamlit session state maintains conversation context

---

## 🛠️ Tech Stack

### Core Technologies
- **Python 3.9+**: Primary language
- **LangChain 0.2.0+**: Agent framework and orchestration
- **OpenAI API**: GPT-4o-mini (reasoning) + GPT-4o (vision)
- **Streamlit**: Web application framework

### Mathematical Libraries
- **SymPy**: Symbolic mathematics (calculus, algebra)
- **NumPy**: Numerical computations
- **Matplotlib**: Data visualization and graphing

### Development & Deployment
- **Git**: Version control
- **Streamlit Cloud**: Free hosting platform
- **Environment Variables**: Secure configuration management

---

## 🚀 Getting Started

### Prerequisites
- Python 3.9 or higher
- OpenAI API key ([Get one here](https://platform.openai.com/api-keys))
- Git (for cloning)

### Local Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/bunyodjonsattorov/math_solver.git
   cd math_solver
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up API key**
   ```bash
   export OPENAI_API_KEY=sk-your-key-here
   ```
   Or create a `.env` file:
   ```bash
   echo "OPENAI_API_KEY=sk-your-key-here" > .env
   ```

5. **Run the application**
   ```bash
   streamlit run app.py
   ```

   The app will open in your browser at `http://localhost:8501`

### Streamlit Cloud Deployment

1. **Push to GitHub** (if not already)
   ```bash
   git add .
   git commit -m "Initial commit"
   git push origin main
   ```

2. **Deploy on Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Click "New app"
   - Connect your GitHub repository
   - Add your `OPENAI_API_KEY` in "Secrets"
   - Deploy!

3. **Configure Secrets**
   In Streamlit Cloud, go to "Settings" → "Secrets" and add:
   ```toml
   OPENAI_API_KEY = "sk-your-key-here"
   ```

---

## 📁 Project Structure

```
math_solver/
├── agent.py          # LangChain agent configuration and prompt engineering
├── app.py            # Streamlit web application (main UI)
├── main.py           # CLI entry point (optional)
├── config.py         # Configuration management (API keys)
├── requirements.txt  # Python dependencies
├── README.md         # This file
└── data/             # PDF textbooks (optional, for future RAG)
```

### Key Files Explained

- **`agent.py`**: Contains the core AI agent logic, including strict prompts that enforce code execution and prevent hallucination
- **`app.py`**: Streamlit UI with chat interface, image upload, and graph display
- **`main.py`**: Command-line interface for testing without the web UI
- **`config.py`**: Centralized API key management from environment variables

---

## 💡 Usage Examples

### Example 1: Integration Problem
**Input**: "Find the integral of 12(2x-5)² + 8x, given that the curve passes through (2, 4)"

**Agent Process**:
1. Recognizes integration problem
2. Writes Python code using SymPy
3. Integrates the function
4. Substitutes point to find constant C
5. Returns final expression

**Output**: Step-by-step solution with Python code visible

### Example 2: Graphing
**Input**: "Plot the graph of y = x² - 4x + 3"

**Agent Process**:
1. Generates matplotlib code
2. Creates plot with proper labels and grid
3. Saves to `graph.png`
4. Displays in UI

**Output**: Visual graph embedded in chat

### Example 3: Image Upload
**Input**: Upload screenshot of a math problem

**Agent Process**:
1. GPT-4o Vision transcribes the problem
2. Agent solves using Python code
3. Returns solution

**Output**: Transcribed problem + solution

---

## 🎓 About the Developer

**Bunyodjon Sattorov**

A second-year Computer Science student with a strong background in mathematics (Math Olympiad experience) and a passion for building practical AI applications. This project demonstrates the intersection of mathematical problem-solving skills and modern software engineering practices.

### Why This Project?

This project was built to:
- **Bridge Theory and Practice**: Apply AI agent frameworks (LangChain) to solve real-world educational problems
- **Demonstrate Full-Stack Skills**: From backend agent development to frontend UI design
- **Show Production Readiness**: Secure API key management, error handling, and deployment
- **Solve a Real Problem**: Help A-Level students learn mathematics through step-by-step solutions

### Technical Skills Demonstrated

- **AI/ML**: LangChain agents, LLM integration, prompt engineering
- **Backend**: Python, agent orchestration, tool integration
- **Frontend**: Streamlit, UI/UX design, state management
- **DevOps**: Git, deployment, environment configuration
- **Mathematics**: Symbolic computation, numerical methods, visualization

---

## 🔮 Future Enhancements

- [ ] **RAG Integration**: Add textbook knowledge base for more contextual solutions
- [ ] **Multi-Language Support**: Support for problems in different languages
- [ ] **Export Solutions**: PDF export with formatted LaTeX
- [ ] **Problem Difficulty Levels**: Adaptive difficulty based on student progress
- [ ] **Voice Input**: Speech-to-text for hands-free problem input
- [ ] **Collaborative Features**: Share solutions with classmates
- [ ] **Performance Analytics**: Track learning progress over time

---

## 🤝 Contributing

Contributions are welcome! This project is open to improvements, especially:
- Enhanced prompt engineering for better accuracy
- Additional mathematical capabilities
- UI/UX improvements
- Performance optimizations

To contribute:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

---

## 🙏 Acknowledgments

- **LangChain Team**: For the excellent agent framework
- **OpenAI**: For GPT-4o and GPT-4o-mini APIs
- **Streamlit**: For the intuitive web framework
- **SymPy Community**: For powerful symbolic mathematics tools

---

## 📧 Contact

**Bunyodjon Sattorov**
- GitHub: [@bunyodjonsattorov](https://github.com/bunyodjonsattorov)
- Project Link: [https://github.com/bunyodjonsattorov/math_solver](https://github.com/bunyodjonsattorov/math_solver)

---

<div align="center">

**Built with ❤️ for students learning A-Level Mathematics**

⭐ Star this repo if you find it helpful!

</div>
