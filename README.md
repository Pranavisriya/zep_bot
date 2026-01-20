# ZEP CHATBOT 

An intelligent, end-to-end conversational AI system built with **Streamlit**, **OpenAI**, and **Zep Cloud** that supports persistent memory, automatic entity extraction, and interactive knowledge graph visualization.
This project enables real-time chat with contextual memory while continuously extracting entities and relationships from conversations and visualizing them as an explorable entity graph. 

## Prerequisites

- Python 3.8 or higher  
- pip or uv  
- API keys for:
  - OpenAI
  - Zep Cloud


## Installation

1. Clone the repository:
```bash
git clone https://github.com/Pranavisriya/zep_bot.git
cd zep_bot
```

2. Install `uv` in the environment if it is not present
```bash
pip install uv
```

3. Create a virtual python environment in this repo
```bash
uv init
uv venv -p 3.12
```

Any other method can also be used to create python environment.

4. Activate python environment
```bash
source .venv/bin/activate
```


5. Install dependencies using uv:
```bash
uv add -r requirements.txt
```

6. Create a `.env` file in the project root with your API keys:
```
ZEP_API_KEY=your_zep_cloud_api_key
OPENAI_API_KEY=your_openai_api_key
```

## Usage

Run the Streamlit application:
```bash
streamlit run app.py
```


## Features

- Real-time conversational chat powered by OpenAI GPT models, providing coherent, context-aware responses that adapt dynamically as the conversation evolves.
- Persistent memory management using Zep Cloud, enabling storage of complete conversation history, support for multiple conversation threads per user, and long-term contextual recall across sessions.
- Automatic extraction of entities such as people, organizations, locations, skills, and key concepts directly from user–assistant interactions without requiring manual annotation.
- Intelligent inference of relationships between extracted entities based on conversational context, allowing the system to build meaningful semantic connections over time.
- Interactive knowledge graph visualization embedded within the Streamlit interface, enabling users to explore entity relationships in real time as the conversation progresses.


  
## License

This project is licensed under the terms included in the LICENSE file.

## Author

Pranavi Sriya (pranavisriyavajha9@gmail.com)






