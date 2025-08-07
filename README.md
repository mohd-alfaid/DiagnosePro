WEB-APP LINK: https://diagnose-pro.streamlit.app/

🧠 DiagnosePro
AI-Powered Brain Tumor Detection Tool

DiagnosePro is a smart and user-friendly web application that uses deep learning to detect brain tumors from MRI scans. Built with Streamlit and powered by a fine-tuned VGG19 neural network, it provides quick and reliable predictions to assist doctors and patients in early screening and awareness.

🔍 Features
🧠 Brain Tumor Detection using MRI images

⚙️ Deep Learning model based on VGG19 architecture

🖼️ Real-time image classification (Tumor / No Tumor)

📄 Informative Articles section with resources about brain tumors

📥 Google Drive integration to download and load pre-trained models

🎨 Visually rich interface with engaging Lottie animations

👨‍⚕️ Doctor recommendation system based on diagnosis result

🧰 Tech Stack
Technology	Usage
Python	Core logic and backend
Streamlit	Web framework for UI
TensorFlow / Keras	Deep learning model
VGG19	Pre-trained CNN used for MRI classification
NumPy & OpenCV	Image preprocessing
gdown	Google Drive model file download
Lottie	Interactive animations
HTML/CSS (via Streamlit)	Basic layout/styling

🧪 How It Works
User uploads an MRI scan via the web interface.

The image is preprocessed and passed through a VGG19-based CNN model.

The model classifies the image into either:

No Tumor

Tumor

Depending on the result:

If No Tumor → Recommends a general physician.

If Tumor → Suggests a specialist doctor.

The user is shown visually appealing animations and guided resources.

📁 Project Structure
bash
DiagnosePro/
│
├── app.py                  # Main Streamlit app logic
├── brain.json              # Lottie animation for diagnosis
├── .streamlit/             # Streamlit config (if any)
├── README.md               # Project documentation
├── requirements.txt        # Python dependencies
└── assets/ (optional)      # Images / Animations / Docs
🧭 Setup Instructions
Clone the Repository

bash
git clone https://github.com/mohd-alfaid/DiagnosePro.git
cd DiagnosePro
Create and Activate a Virtual Environment

bash
 
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
Install Dependencies

bash
 
pip install -r requirements.txt
Run the App

bash
 
streamlit run app.py
Note: The app automatically downloads the model files from Google Drive when needed.

🔗 Live Demo (Optional)
Coming soon...
You can deploy this app on platforms like Streamlit Cloud, Render, or Heroku for public access.

✍️ Contributing
Contributions are welcome!
Feel free to fork this repo, create a new branch, and submit a pull request.

If you find any issues or want to suggest improvements, open an issue here.

⚖️ License
This project is licensed under the MIT License – feel free to use and modify it.

🙏 Acknowledgements
LottieFiles for animations

Keras & TensorFlow for model training

Streamlit for making UI effortless

Open-source MRI datasets for model development

Medical experts and advisors who guided the problem framing


<img width="1280" height="699" alt="image" src="https://github.com/user-attachments/assets/68f7ac00-0bba-4dd7-b1a3-0ebb91f01449" />

