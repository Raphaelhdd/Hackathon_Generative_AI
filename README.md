# Hackathon Lightricks 2023 - Startup Incubator Tool

This project was developed as part of the Lightricks Hackathon 2023, where our team earned 3rd place out of 50 teams. The goal of the project was to build an AI-driven startup incubator tool that provides intelligent suggestions for startup ideas and helps entrepreneurs navigate through various development stages.

## Project Overview

The Startup Incubator Tool uses **OpenAI** for natural language processing and **Firebase** for real-time data storage and management. The tool allows users to input startup ideas, and based on the information provided, it generates detailed business development suggestions and prompts for further research.

### Key Features:
- **AI-driven Prompts:** Generates suggestions for startup ideas by leveraging OpenAI’s language models.
- **Firebase Integration:** Real-time data storage and management, ensuring prompt responses and easy collaboration.
- **Dynamic Recommendation System:** Uses a step-by-step business development approach to guide users through the process of turning their idea into a viable product.

## Technologies Used
- **OpenAI (GPT-3.5)**: For generating intelligent business development suggestions.
- **Firebase**: For handling real-time data, user interaction, and data storage.
- **Python**: Backend development using Python for the integration of OpenAI and Firebase.
- **Cloud Functions**: For event-driven architecture and seamless interaction between components.

## Setup Instructions

To run the project locally, follow the instructions below:

1. **Clone the repository:**
    ```bash
    git clone https://github.com/yourusername/startup-incubator-tool.git
    ```

2. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3. **Set up Firebase:**
    - Create a Firebase project and download your service account credentials in JSON format.
    - Place the JSON file in the project directory and set the environment variable:
    ```bash
    export GOOGLE_APPLICATION_CREDENTIALS="path/to/your/firebase/credentials.json"
    ```

4. **Set up OpenAI API Key:**
    - Obtain an API key from OpenAI.
    - Add it to your environment variables:
    ```bash
    export OPENAI_API_KEY="your-api-key"
    ```

5. **Run the application:**
    ```bash
    python main.py
    ```

## Usage

Once the project is set up, you can interact with the tool by submitting startup ideas. The system will respond with AI-generated business development steps and suggestions for further exploration.

## Contribution

Feel free to fork the repository, submit issues, and create pull requests if you'd like to contribute to the project!

## Acknowledgments

- Thanks to **Lightricks** for hosting the hackathon and providing us with the opportunity to participate.
- A special thanks to the **OpenAI** team for providing the API that powered the AI-driven suggestions.
