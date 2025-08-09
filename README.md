# Face Recognition System

A Python-based **Face Recognition System** using `dlib` and `OpenCV` that can detect and recognize faces in real time from a webcam feed.  
The system stores known faces' encodings in a `data.json` file and can identify them when they appear.

---

## 🚀 Features
- Real-time **face detection** from webcam
- **Face recognition** using `dlib` face encodings
- Store and manage **known persons** in `data.json`
- **Automatic labeling** of recognized faces
- Works with **multiple people** at the same time
- Supports **adding new faces** easily

---

## 📂 Project Structure
Face-Recognition-System/
│── data.json # Stores known persons' face encodings & names
│── main.py # Main program for real-time recognition
│── encode_faces.py # Script to add new faces to the database
│── requirements.txt # Required Python libraries
│── README.md # Project documentation



---

## 🛠 Installation

### 1️⃣ Clone the repository

git clone https://github.com/yourusername/Face-Recognition-System.git
cd Face-Recognition-System

2️⃣ Install dependencies
Make sure you have Python 3.8+ installed, then run:


pip install -r requirements.txt
requirements.txt


opencv-python
dlib
face-recognition
numpy

▶ Usage
1. Add a new person
Run the following to capture a new face and save it in data.json:


python encode_faces.py
Enter the name when prompted

Look at the camera until capture completes

2. Start face recognition
python main.py
The webcam will open

Recognized faces will be labeled with names

Unknown faces will be marked as "Unknown"

🖼 Example Output
Terminal

[INFO] Loading known faces...
[INFO] Starting webcam...
[INFO] Recognized: John Doe
Webcam Window

Green box around known faces with name label

Red box around unknown faces labeled "Unknown"

⚡ How It Works
Face Detection → Uses dlib’s HOG + CNN-based model to find faces.

Face Encoding → Extracts a unique 128-dimension vector for each face.

Matching → Compares the encoding with stored encodings in data.json.

Real-Time Display → Shows results on webcam feed.

📌 Future Improvements
Store face images along with encodings

Improve accuracy using deep learning models

Add GUI for easy interaction

Support multiple camera inputs

📜 License
This project is licensed under the MIT License - you are free to use and modify it.

🤝 Contributing
Pull requests are welcome!
For major changes, open an issue first to discuss what you’d like to change.

💡 Author
Uveash Reangrezz
📧 Email: rangrezzuveash02@gmail.com
🌐 GitHub: uveash-rangrezz02


---

I also suggest we include **a setup image or screenshot** of your face recognition system running — it makes your README look more professional on GitHub.  

Do you want me to add **a sample screenshot section** in this README? That will make it visually appealing.
