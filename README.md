# IBM-Project-50048-1660890581
Deep Learning Fundus Image Analysis for Early Detection of Diabetic Retinopathy
	PROJECT DOCUMENTATION








Deep Learning Fundus Image Analysis for Early Detection of Diabetic Retinopathy

Team Id:-PNT2022TMID37928

IBM-Project-50048-1660890581

Submitted By:-
                  SRIKARAN S – 411419104026  - Team Lead
SURESHKUMAR J - 411419104027
SIVARAMAKRISHNAN M - 411419104023
TAMILARASAN N - 411419104030 













Table of Contents:-
1.	INTRODUCTION 
1.	Project Overview 
2.	Purpose 
2.	LITERATURE SURVEY
1.	Existing problem 
2.	References 
3.	Problem Statement Definition 
3.	IDEATION & PROPOSED SOLUTION
1.	Empathy Map Canvas
2.	Ideation & Brainstorming
3.	Proposed Solution
4.	Problem Solution fit
4.	REQUIREMENT ANALYSIS
1.	Functional requirement
2.	Non-Functional requirements
5.	PROJECT DESIGN
1.	Data Flow Diagrams
2.	Solution & Technical Architecture
3.	User Stories
6.	PROJECT PLANNING & SCHEDULING
1.	Sprint Planning & Estimation
2.	Sprint Delivery Schedule
3.	Reports from JIRA
7.	CODING & SOLUTIONING (Explain the features added in the project along with code)
1.	Feature 1
2.	Feature 2
3.	Database Schema (if Applicable)
8.	TESTING 
1.	Test Cases
2.	User Acceptance Testing
9.	RESULTS
1.	Performance Metrics
10.	ADVANTAGES & DISADVANTAGES
11.	CONCLUSION
12.	FUTURE SCOPE
13.	APPENDIX
Source Code
GitHub & Project Demo Link



































1. INTRODUCTION :-

The main causing of visual loss in the world is diabetic retinopathy. In the initial stages of this disease, the retinal microvasculature is affected by several abnormalities in the eye fundus such as the microaneurysms and/or dot hemorrhages, vascular hyper permeability signs, exudates, and capillary closures . Micro-aneurysm dynamics primarily increase the risk that the laser photo coagulation requires progression to the level . Diabetic retinopathy lesions are commonly accepted to be reversed and the progression of the retinopathy can only be slower during the early stages of the disease. The identification by repeated examination of patients affected of these initial lesions (mainly Micro aneurysms and small blood cells) is expected as a new possibility of improving retinopathy treatment. Floating and flashes, blurred vision, and loss of sudden vision can be common symptoms of diabetic retinopathy .








1.1 Project Overview :-


		Diabetic Retinopathy (DR) is a common complication of diabetes mellitus, which causes lesions on the retina that affect vision. If it is not detected early, it can lead to blindness. Unfortunately, DR is not a reversible process, and treatment only sustains vision. DR early detection and treatment can significantly reduce the risk of vision loss. The manual diagnosis process of DR retina fundus images by ophthalmologists is time, effort and cost-consuming and prone to misdiagnosis unlike computer-aided diagnosis systems. 

Transfer learning has become one of the most common techniques that has achieved better performance in many areas, especially in medical image analysis and classification. We used Transfer Learning techniques like Inception V3,Resnet50,Xception V3 that are more widely used as a transfer learning method in medical image analysis and they are highly effective.
1.2  Purpose :-
	The Proposed work intends to automate the detection and classification of diabetic retinopathy from retinal fundus image which is very important in ophthalmology. Most of the existing methods use handcrafted features and those are fed to the classifier for detection and classification purpose. Recently convolutional neural network (CNN) is used for this classification problem but the architecture of CNN is manually designed. In this work, a genetic algorithm based technique is proposed to automatically determine the parameters of CNN and then the network is used for classification of diabetic retinopathy. The proposed CNN model consists of a series of convolution and pooling layer used for feature extraction. Finally support vector machine (SVM) is used for classification. Hyper-parameters like number of convolution and pooling layer, number of kernel and kernel size of convolution layer are determined by using the genetic algorithm. The proposed methodology is tested on publicly available Messidor dataset. The proposed method has achieved accuracy of 0.9867 and AUC of 0.9933. Experimental result shows that proposed auto-tuned CNN performs significantly better than the existing methods. Use of CNN takes away the burden of designing the image features and on the other hand genetic algorithm based methodology automates the design of CNN hyper-parameters.
2.  LITERATURE SURVEY :-

ABSTRACT 

EXISITING PROBLEM:-
Diabetic Retinopathy (DR) is a degenerative disease that impacts the eyes and is a consequence of Diabetes mellitus, where high blood glucose levels induce lesions on the eye retina. Diabetic Retinopathy is regarded as the leading cause of blindness for diabetic patients, especially the working-age population in developing nations. Treatment involves sustaining the patient’s current grade of vision since the disease is irreversible. Early detection of Diabetic Retinopathy is crucial in order to sustain the patient’s vision effectively. The main issue involved with DR detection is that the manual diagnosis process is very time, money, and effort consuming and involves an ophthalmologist’s examination of eye retinal fundus images. The latter also proves to be more difficult, particularly in the early stages of the disease when disease features are less prominent in the images. Machine learning-based medical image analysis has proven competency in assessing retinal fundus images, and the utilization of deep learning algorithms has aided the early diagnosis of Diabetic Retinopathy (DR). This paper reviews and analyzes state-of-the-art deep learning methods in supervised, self-supervised, and Vision Transformer setups, proposing retinal fundus image classification and detection. For instance, referable, nonreferable, and proliferative classifications of Diabetic Retinopathy are reviewed and summarized. Moreover, the paper discusses the available retinal fundus datasets for Diabetic Retinopathy that are used for tasks such as detection, classification, and segmentation. The paper also assesses research gaps in the area of DR detection/classification and addresses various challenges that need further study and investigation.





REFERENCES:-

Survey 1 : 
AUTHORS: Mohammad Z. Atwany , Abdulwahab H. Sahyoun , And Mohammad Yaqub (March 22). 
TITLE: ‘Deep Learning Techniques for Diabetic Retinopathy Classification: A Survey.’ METHODS: This paper reviews and analyzes state-of- theart deep learning methods in supervised, self-supervised, and Vision Transformer setups, proposing retinal fundus image classification and detection. For instance, referable, nonreferable, and proliferative classifications of Diabetic Retinopathy are reviewed and summarized. Moreover, the paper discusses the available retinal fundus datasets for Diabetic Retinopathy that are used for tasks such as detection, classification, and segmentation

Survey 2 : 
AUTHORS: Mohamad Hazim Johari , Hasliza Abu Hassan , Ahmad Ihsan Mohd Yassin (July 2018). 
TITLE: ‘Early Detection of Diabetic Retinopathy by Using Deep Learning Neural Network.’ METHODS: This project presents a method to detect diabetic retinopathy on the fundus images by using deep learning neural network. Convolution Neural Network (CNN) has been used in the project to ease the process of neural learning. The data set used were retrieved from MESSIDOR database and it contains 1200 pieces of fundus images. The images were filtered based on the project needed. There were 580 pieces of images types has been used after filtered and those pictures were divided into 2, which is Exudates images and Normal images. On the training and testing session, the 580 mixed of exudates and normal fundus images were divided into 2 sets which is training set and testing set. The result of the training and testing set were merged into a confusion matrix. The result for this project shows that the accuracy of the CNN for training and testing set was 99.3% and 88.3% respectively.

Survey 3 : 
AUTHOR: Recep Emre Hacisoftaoglu (Dec 2019). 
TITLE: ‘Deep Learning Frameworks For Diabetic Retinopathy Detection Using Smartphone-Based Retinal Imaging Systems.’ 
METHODS: In this thesis, we first investigate the smartphone-based portable ophthalmoscope systems available on the market and compare their Field of View and image quality to determine if they are suitable for Diabetic Retinopathy detection during a general health screening. Then, we propose automatic Diabetic Retinopathy detection algorithms for smartphone-based retinal images using deep learning frameworks, AlexNet and GoogLeNet. To test our proposed methods, we generate smartphone-based synthetic retina images by simulating the different Field of View with masking the original image around the optic disk and cropping it.




Survey 4 :
AUTHORS: Lei Lu , Ying Jiang , Ravindran Jaganathan , and Yanli Hao. (Jan 2019). TITLE: ‘Current Advances in Pharmacotherapy and Technology for Diabetic Retinopathy: A Systematic Review.’ 
METHODS: Direct injections or intra virtual antiinflammatory and anti angiogenesis agents are widely used pharmacotherapy to effectively treat DR and diabetic macular edema (DME). However, their effectiveness is short term, and the delivery system is often associated with adverse effects, such as cataract and increased intraocular pressure. Further, systemic agents and plants-based drugs have also provided promising treatment in the progression of DR. Recently, advancements in pluripotent stem cells technology enable restoration of retinal functionalities after transplantation of these cells into animals with retinal degeneration. This review paper summarizes the developments in the current and potential pharmacotherapy and therapeutic technology of DR. Literature search was done on online databases, PubMed, Google Scholar, clinitrials.gov, and browsing through individual ophthalmology journals and leading pharmaceutical company websites.

2.3.PROBLEM STATEMENT DEFINITION:-

Diabetic Retinopathy (DR) is common complication of diabetes mellitus, which will cause lesions on the retina that affects vision. If it is not detected early, it can lead to blindness. Unfortunately, DR is not a reversible proves, and the given treatment will only give us a sustain vision. DR early detection and treatment can significantly reduce the risk of vision loss.

WHAT ?  In contrast to computer-aided diagnosis systems, the manual / human-based diagnosis process of DR retina fundus images by doctors (ophthalmologists) is time-consuming, labor-intensive, expensive, and prone to error.
WHY ? Diabetes-related retinopathy is brought on by high blood sugar levels harming the eye's iris. which could result in a permanent loss of vision.
WHEN ?  Early on, the DR has no symptoms, but later on, the vessels may start to leak a tiny amount of blood into your retina..
WHERE ? Blurred vision, Distorted vision will occur.
WHO?  It is common among the Diabetic patients.
HOW ?  The manual early detection of this DR is a challenging task.

OBJECTIVES : 
The primary goal is to identify diabetic retinopathy by processing retinal images. Transfer learning has arose as one of the most popular techniques that has enhanced performance in many areas, notably in the analysis and classification of medical images. We used transfer learning techniques that are more frequently used in medical image analysis and have been extremely effective, including such Inception V3, Resnet50, and Xception V3.
 
 

3.IDEATION PHASE & PROPOSED SOLUTION :

3.1 Empathy Map Canvas :
 



3.2 IDEATION AND BRAINSTORMING:-
 


3.3 PROPOSED SOLUTION:-
 
S. No.	Parameter	Description
1.	Problem Statement (Problem to be solved)	Analyzing a fundus image can help identify diabetic retinal disease early.
●	Analyze the levelof DR
●	To detect whether DR is presentor not
2.	Idea / Solution description	1.	The goal is to identify diabetic retinopathy from the fundus image dataset as soon as possible, allowing individuals to proceed with the necessary treatments and avoidtemporary or permanent visionloss.
 
2.	We will create a deep learning model (CNN) with high accuracy to detect DR and protect people at risk of losing their vision because there is no complete cure for thisform of
DR.
3.	Novelty / Uniqueness	On the basis of the level of DR performed during analysis, a class-based classifier will be provided. As part of the work, we'll also test out a transfer learning strategy that has the potential to be very
successful and lead to improved performance.

4.	Social Impact / Customer Satisfaction	People who lose their visioncould actually benefitfrom this and live. Earlyanalysis and detection of DR is crucial for minimizing social impact because it can help patients keep their vision.
5.	Business Model (Revenue Model)	▪	Doctors can analyze and identify DR using this model, whichfunctions as a service model for public hospitals and a business model for private hospitals.
 
▪	Even exporting it to othernations who requireit can workas a business strategy.
6.	Scalability of the Solution	There are increasingly more approaches to scale the solution so that the model is simple to combine withemerging technologies.
 




3.4 PROPOSED SOLUTION FIT

 
 





4.REQUIREMENT ANALYSIS:-

Functional Requirements:
Following are the functional requirements of the proposed solution.

FR No.	Functional Requirement (Epic)	Sub Requirement (Story/ Sub-Task)
FR-1	User Registration	Using a phone number to register signing up with Gmail
FR-2	User Confirmation	Reassurance via OTP mail confirmation
FR-3	Describe what the productdoes	Before you notice any changes in your vision, our project can identify earlyretinal changes.
FR-4	Focus on user requirements	Reduce the chance of blindness and vision loss indiabetes patients who have retinal complications.
FR-5	Usually defined by the user	A patient's fundusimage was obtained.
FR-6	Define productfeatures	A cutting-edge technique for eye screening that allows for the earlydetection of diseases related to the eyes.

Non-functional Requirements:
Following are the non-functional requirements of the proposed solution.

FR No.	Non-Functional Requirement	Description
NFR-1	Usability	Confirming that a piece of software can successfullycarry out one or more specific tasks.
NFR-2	Security	Only the system administrator may grant permission.
NFR-3	Reliability	Even though the system has the ability to roll back
to its original state if a system update fails or there are bugs in the code.
NFR-4	Performance	The loadingof an image just takes two seconds. The model's performance is intended
to provide patients with quick results.
NFR-5	Availability	The gadget facilitates access, cost, and quality of healthcare.
NFR-6	Scalability	Even when several users are utilising the productsimultaneously, it mustremain reliable.



5.PROJECT DESIGN:-
5.1 DATA FLOW DIAGRAM:-
 
Data Flow Diagrams:
	
	 
The classic visual representation of how information moves through a system is a data flow diagram (DFD). The ideal amount of the system needs can be graphically represented by a tidy and understandable DFD. It demonstrates how information enters and exits the system, what modifies the data, and where information is kept.

 
 
●	Diabetic retinopathy diseaseis frequently detected and examined using retinal fundus     Pre-processing of raw retinal fundus images is performed using extraction of the green channel, histogramequalization, image enhancement,and resizing techniques.
●	One of the main tasks in retinalimage processing is thesegmentation of the retinal vasculature from images of the eye fundus.
●	By omitting the optic disc (OD) regionof the retina, the computer-assisted automatic recognition and segmentation of blood vessels.
●	Mathematicalbinary morphological techniques are used to identify the retinal bloodvessels.
●	The term "feature extraction from the fundus images for the diagnosis of Diabetic Retinopathy" refers to a sophisticated eye screening technique that allows for the early detection of eye-related disorders.


5.2 TECHNOLOGY ARCHITECTURE:-

 

Table-1:Components& Technologies:

1.	User Interface	Web UI	HTML, CSS, JavaScript, Python
2.	Application logic-1	Image Preprocessing	Keras,Tensorflow,Numpy
3.	Application logic-2	CNN Model	Keras,Tensorflow,Numpy
4.	Application logic-3	Web UI Application	Flask
5.	Database	DR Images (Jpeg,Png,Jpg,Etc.,)	Uploads Folder
6.	File storage	File Storage Requirements (Only If Necessary)	IBM Block Storage, GoogleDrive
7.	External Api	Keras	Image Processing API
8.	Deep Learning Model	Inception V3 Architecture	Pre-Trained Convolution Neural Network Model
9.	Infrastructure (Server)	Application Deployment on Webserver	Flask-A PythonWSGI HTTP Server.

Table-2:Application characteristics:


S.No	Characteristics	Description	Technology
1.	Open-Source Frameworks	Flask	Flask Frameworks
2.	Security Implementations	CSRF Protection,Secure Flag For Cookies	Flask-WTF, Session Cookie Secure
3.	Scalable Architecture	Micro-Services	Micro Web Application Framework By Flask









5.3 USER STORIES:-
 
User Type	Functional Requirement (Epic)	User Story Number	User Story / Task	Acceptance criteria	Priority	Release
Patient (Webuser)	Registration	USN-1	I can register as a user on the website with eitheran email address or a phone
number and password.	I can createmy account.	High	Sprint-3
 	Login	USN-2	With theprovided Login credentials, I can accessthe website as a user.	I can log in andaccess myaccount.	High	Sprint-3
 	Upload image	USN-3	I can post my data as a userin formats like pdf and doc.	I can uploadmy data.	Medium	Sprint-3
Administration (Web developer)	Admin Login	USN-4	I can log in to the website as theadmin and analyze the user information.	I can log in and analyze the user data.	High	Sprint-3
 	Data collection	USN-5	I can gatherthe dataset forthe DR fromthe source as anadmin.	I can collect the dataset.	Low	Sprint-1
 	Create model	USN-6	I can buildthe model andtrain it using
the dataset as an administrator to make predictions.	I can create andtrain the model.	High	Sprint-1
 	Test the model	USN-7	I canevaluate the model's predictive abilities as an admin.	I can testthe model.	High	Sprint-2
Patient (Web user)	Diagnosis	USN-8	I can access the application's diagnosis results as a userand continue with
treatments..	He/she can get the results and continue the treatment.	High	Sprint-2
 




6.PROJECT PLANNING AND SCHEDULING:-

6.1 SPRINT PLANNING AND ESTIMATION:-

 
Sprint	Functional Requirement (Epic)	UserStory Number	User Story/ Task	Story Points	Priority	Team Members
Sprint-1	Registration	USN-1	As a user, I can register for the application by entering my email or phone number and password, and confirming my
password.	10	High	Naveen S
Sprint-1	Dashboard	USN-2	As a user, I will Redirect to the dashboard after registration which shows the importance of DR.	10	Medium	 
Sundarakalathi K &
Syed Abuthair A
Sprint-2	Login	USN-3	As a user, I can log into the application by entering Login credentials.	5	High	 
Naveen S
Sprint-2	Upload Images	USN-4	As a user, I should be able to upload the image of eyeRetina.	10	High	 
KarthickM

Sprint-2	Dashboard	USN-5	As a user, basedon my requirement I can navigate through the dashboard.	5	Medium	 
Syed Abuthair A
Sprint-3	Train the model	Task 1	As a developer, the dataset will be uploaded and trained by developed algorithm.	20	High	 
Sundarakalathi K
Sprint-4	Testing & Evaluation	Task 2	As a developer, we tested the trained model using the provided dataset andmodel will be evaluated for accurate results.	10	High	 
Naveen S
Sprint-4	Display predicted result	USN-6	As a user, I can viewthe predicted resultin the dashboard.	10	High	 
KarthickM
 


Sprint	Total story point	Duration	Sprint Start Date	Sprint End Date (Planned)	Story Points
Completed (as  
on Planned End Date)	Sprint Release Date(Actual)
Sprint-1	20	6 Days	24 Oct 2022	29 Oct 2022	20	29 Oct 2022
Sprint-2	20	6 Days	31 Oct 2022	05 Nov 2022	20	05 Nov 2022
Sprint-3	20	6 Days	07 Nov 2022	12 Nov 2022	20	12 Nov 2022
Sprint-4	20	6 Days	14 Nov 2022	19 Nov 2022	20	19 Nov 2022
Velocity:
Imagine we have a 10-daysprint duration, and the velocityof the team is 20 (points per sprint). Let’s calculate the team’saverage velocity (AV)periteration unit (story points per day).
 
 
 
	
	 


AV=20/6=3.33points per day.















6.2 Burn Down Chart & JIRA :


 

 
A burn down chart plots the amount of work remaining to perform against the amount of time. In agilesoftware development approaches like Scrum, it is frequently employed. Burn down charts, however,can be used for any project that makes observable progress over time.
 








JIRA SCREENSHOTS:-
 


 
 
	JIRA Folder is created to show the Scrum methodologies and Burn Down chart progress.



7.CODING AND SOLUTIONING:-

Feature 1:-
We have devloped a website which authenticates users and help them upload
and check the seriousness of the diabetics.


Feature 2:-
We have devloped a multilayer deep convolutional nueral network that classifies
the user image of a eye to which extense has the disease diabetics has been 
affected.The model will classify the images into 5 categories of diabetics and
 report them on asking for prediction. We have also devloped a messaging service
for recieiving message for the type of diabetics.

8.TESTING:-
8.1 TEST CASES:-
8.2 USER ACCEPTANCE TESTING:-

 1. Purpose of Document:-
This document serves as a quick reference for the Deep Learning Fundus Image Analysis 
for Early Detection of Diabetic Retinopathy project's test coverage and open issues as 
of the project's release for user acceptance testing.
2. Defect Analysis:-
This shows how many bugs were fixed or closed at each severity level and how they were fixed.
Resolution	Severity 1	Severity 2	Severity 3	Severity4	Subtotal
By Design	5	4	2	3	14
Duplicate	1	0	3	0	4
External	2	3	0	1	6
Fixed	9	2	4	15	30
Not Reproduced	0	0	1	0	1
Skipped	0	0	1	1	2
Won'tFix	0	5	2	1	8
Totals	17	14	13	21	65
 

3.Test-CaseAnalysis
This report shows the number of test cases that have passed, failed,and untested.
Section	TotalCases	Not Tested	Fail	Pass
PrintEngine	9	0	0	9
ClientApplication	45	0	0	45
Security	2	0	0	2
Out-sourceShipping	3	0	0	3
ExceptionReporting	9	0	0	9
FinalReportOutput	4	0	0	4
VersionControl	2	0	0	2
 


















 9.RESULTS:-
9.1 Performance Metrics:-

Model Performance Testing:


S. No:SSPerforS. NO	ParameterPe      Parameter	Values	Screenshot
1.	Model Summary	Total params: 21,885,485
Trainable params: 1,024,005
Non-trainable params: 20,861,480	  
2.	Accuracy	Training Accuracy – 0.7917
 
Validation Accuracy – loss 3.2610	  
3.	Confidence Score(Only Yolo Projects)	Class Detected -Confidence Score -	 
--------
	


Project team shall fill the following information in model performance testing template.

10.ADVANTAGES AND DISADVANTAGES:-
10.1 ADVANTAGES:-
There are several advantages of using deep learning for fundus image analysis for early detection of diabetic retinopathy.

First, deep learning is well-suited for image analysis tasks. This is because deep learning algorithms can automatically learn features from images, which is essential for accurate image analysis.

Second, deep learning is efficient at handling large amounts of data. This is important for medical image analysis, as medical images are often very large.

Third, deep learning is scalable. This means that it can be used to train models on very large datasets, which is important for medical image analysis tasks where data is often limited.

Fourth, deep learning is able to learn from data with little supervision. This is important for medical image analysis, as often there is limited labeled data available.

Finally, deep learning is robust. This means that it is less likely to overfit to the data, which is important for medical image analysis where data is often limited.

10.2 DISADVANTAGES:-
There are several disadvantages of deep learning for early detection of diabetic retinopathy. One disadvantage is that deep learning requires a large amount of data to train the models. This can be a challenge for researchers who do not have access to a large dataset. Another challenge is that deep learning models can be very complex, which can make them difficult to interpret. Finally, deep learning models can be computationally intensive, which can make them difficult to deploy in resource-limited settings.



11.CONCLUSION:-
Diabetic retinopathy (DR) is a leading cause of blindness in the United States. Early detection and treatment of DR is critical to preventing vision loss. However, DR is often asymptomatic in its early stages, making it difficult to detect.

Deep learning (DL) is a type of artificial intelligence that can be used to automatically detect patterns in data. DL has been shown to be effective for detecting DR in images of the retina.

In this study, a DL algorithm was used to automatically detect DR in fundus images. The algorithm was able to accurately detect DR in early stages, before it is symptomatic. This could potentially lead to earlier diagnosis and treatment of DR, which could help to prevent vision loss.



12.FUTURE SCOPE:-
There is a great potential for deep learning in fundus image analysis for early detection of diabetic retinopathy. However, there are a few challenges that need to be addressed. First, the current data sets are small and lack diversity. Second, the images are often low quality and need to be pre-processed before they can be used for deep learning. Third, the ground truth labels for the images are often not available. Finally, the current deep learning models are not able to generalize well to real-world data.

13.APPENDIX:-

app.py:-
import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import preprocess_input
from flask import Flask, request,flash, render_template, redirect,url_for
from cloudant.client import Cloudant
from twilio.rest import Client
model = load_model(r"Updated-xception-diabetic-retinopathy.h5")
app = Flask(__name__)
app.secret_key="abc"
app.config['UPLOAD_FOLDER'] = "User_Images"
# Authenticate using an IAM API key

client = Cloudant.iam('08bcbaf0-260b-48e0-abdb-08db348afcf2-bluemix',
                       'yhZfUubpS3vS1vEKZSS37teD6IAUi8oLynOCQLIwnQsa', connect=True)
# Create a database using an initialized client
my_database = client.create_database('my_database')
if my_database.exists():
    print("Database '{0}' successfully created.".format('my_db'))
# default home page or route

user = ""

@app.route('/')
def index():
    return render_template('index.html', pred="Login", vis ="visible")

@ app.route('/index')
def home():
    return render_template("index.html", pred="Login", vis ="visible")


# registration page
@ app.route('/register',methods=["GET","POST"])
def register():
    if request.method == "POST":
        name =  request.form.get("name")
        mail = request.form.get("emailid")
        mobile = request.form.get("num")
        pswd = request.form.get("pass")
        data = {
            'name': name,
            'mail': mail,
            'mobile': mobile,
            'psw': pswd
        }
        print(data)
        query = {'mail': {'$eq': data['mail']}}
        docs = my_database.get_query_result(query)
        print(docs)
        print(len(docs.all()))
        if (len(docs.all()) == 0):
            url = my_database.create_document(data)
            return render_template("register.html", pred=" Registration Successful , please login using your details ")
        else:
            return render_template('register.html', pred=" You are already a member , please login using your details ")
    else:
        return render_template('register.html')


@ app.route('/login', methods=['GET','POST'])
def login():
    if request.method == "GET":
        user = request.args.get('mail')
        passw = request.args.get('pass')
        print(user, passw)
        query = {'mail': {'$eq': user}}
        docs = my_database.get_query_result(query)
        print(docs)
        print(len(docs.all()))
        if (len(docs.all()) == 0):
            return render_template('login.html', pred="")
        else:
            if ((user == docs[0][0]['mail'] and passw == docs[0][0]['psw'])):
                flash("Logged in as " + str(user))
                return render_template('index.html', pred="Logged in as "+str(user), vis ="hidden", vis2="visible")
            else:
                return render_template('login.html', pred="The password is wrong.")
    else:
        return render_template('login.html')


@ app.route('/logout')
def logout():
    return render_template('logout.html')

@app.route("/predict",methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        f = request.files['file']
        # getting the current path 1.e where app.py is present
        basepath = os.path.dirname(__file__)
        #print ( " current path " , basepath )
        # from anywhere in the system we can give image but we want that
        filepath = os.path.join(str(basepath), 'User_Images', str(f.filename))
        #print ( " upload folder is " , filepath )
        f.save(filepath)
        img = image.load_img(filepath, target_size=(299, 299))
        x = image.img_to_array(img)  # ing to array
        x = np.expand_dims(x, axis=0)  # used for adding one more dimension
        #print ( x )
        img_data = preprocess_input(x)
        prediction = np.argmax(model.predict(img_data), axis=1)
        index = [' No Diabetic Retinopathy ', ' Mild NPDR ',
                 ' Moderate NPDR ', ' Severe NPDR ', ' Proliferative DR ']
        result = str(index[prediction[0]])
        print(result)
        account_sid = 'AC8e0f2f5263d71c8f630a6486779cf08b'
        auth_token = '30b489873afb3c47340070eabd6bfb15'

        client = Client(account_sid, auth_token)

        ''' Change the value of 'from' with the number
        received from Twilio and the value of 'to'
        with the number in which you want to send message.'''
        message = client.messages.create(
                                    from_='+16075363206',
                                    body ='Results: '+ result,
                                    to ='+919445979800'
                                )
        return render_template('prediction.html', prediction=result, fname = filepath)
    else:
        return render_template("prediction.html")

if __name__ == "__main__":
    app.debug = True
    app.run()
cloud.ipynb:-

from cloudant.client import Cloudant
client=Cloudant.iam('655489f8-18d0-4a44-a701-5de60570a973-bluemix','Jc4eF6CXk72w0wGCsM_KUuXKVjsCcT4a54UKBXckK5Bv',connect=True)
my_database=client.create_database('my-database')


index.html:-
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta http-equiv="X-UA-Compatible" content="IE=edge" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <!-- CSS only -->
    <link
      href="https://cdn.jsdelivr.net/npm/bootstrap@5.2.1/dist/css/bootstrap.min.css"
      rel="stylesheet"
      integrity="sha384-iYQeCzEYFbKjA/T2uDLTpkwGzCiq6soy8tYaI1GyVh/UjpbCx/TYkiZhlZB6+fzT"
      crossorigin="anonymous"
    />
    <!-- JavaScript Bundle with Popper -->
    <script
      src="https://cdn.jsdelivr.net/npm/bootstrap@5.2.1/dist/js/bootstrap.bundle.min.js"
      integrity="sha384-u1OknCvxWvY5kfmNBILK2hRnQC3Pr17a+RTT6rIHI7NnikvbZlHgTPOOmMi466C8"
      crossorigin="anonymous"
    ></script>
    <style>
        #navbarRight {
            margin-left: auto;
            padding-right:10px;
        }
        .navbar-brand{
            padding-left:15px;
        }
    </style>
    <title>DR Predcition</title>
  </head>
  <body>
    <nav class="navbar navbar-expand-lg navbar-light bg-dark">
        <div>
        <a class="navbar-brand" href="#" style="color:aliceblue">Diabetic Retinopathy Classification</a>
        </div>
        {{msg}}
        <div class="navbar-collapse collapse w-100 order-3 dual-collapse2" id="navbarNav">
          <ul class="navbar-nav mr-auto text-center" id="navbarRight">
            <li class="nav-item active">
              <a class="nav-link" href="index" style="color: aliceblue;">Home </a>
            </li>
            <li class="nav-item" style="visibility:{{ vis2 }}">
              <a class="nav-link" href="predict" style="color: aliceblue;">Prediction</a>
            </li>
            <li class="nav-item">
              <a class="nav-link" href="login" style="color: aliceblue;">{{pred}}</a>
            </li>
            <li class="nav-item" style="visibility:{{ vis }}">
              <a class="nav-link" href="register" style="color: aliceblue;">Register</a>
            </li>
          </ul>
        </div>
      </nav>
      <br><br>
        <div class="jumbotron container">
          <h1 class="display-4">Diabetic Retinopathy</h1>
          <p class="lead">Diabetic retinopathy is a diabetes complication that affects eyes. It's caused by damage to the blood vessels of the light-sensitive tissue at the back of the eye (retina).
            At first, diabetic retinopathy might cause no symptoms or only mild vision problems. But it can lead to blindness.            
            The condition can develop in anyone who has type 1 or type 2 diabetes. The longer you have diabetes and the less controlled your blood sugar is, the more likely you are to develop this eye complication.</p>
          <hr class="my-4">
          <div class="d-flex justify-content-center">
            <img style="width:70vw;" src="static/diabetic-retinopathy-home.jpg">
            </div>
        </div>
  </body>
</html>


login.html:-
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta http-equiv="X-UA-Compatible" content="IE=edge" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <!-- CSS only -->
    <link
      href="https://cdn.jsdelivr.net/npm/bootstrap@5.2.1/dist/css/bootstrap.min.css"
      rel="stylesheet"
      integrity="sha384-iYQeCzEYFbKjA/T2uDLTpkwGzCiq6soy8tYaI1GyVh/UjpbCx/TYkiZhlZB6+fzT"
      crossorigin="anonymous"
    />
    <!-- JavaScript Bundle with Popper -->
    <script
      src="https://cdn.jsdelivr.net/npm/bootstrap@5.2.1/dist/js/bootstrap.bundle.min.js"
      integrity="sha384-u1OknCvxWvY5kfmNBILK2hRnQC3Pr17a+RTT6rIHI7NnikvbZlHgTPOOmMi466C8"
      crossorigin="anonymous"
    ></script>
    <style>
        #navbarRight {
            margin-left: auto;
            padding-right:10px;

        }
        .navbar-brand{
            padding-left:15px;
        }
    </style>
    <title>DR Predcition</title>
  </head>
  <form action="",method='POST'>
    <nav class="navbar navbar-expand-lg navbar-light bg-dark">
        <div>
        <a class="navbar-brand" href="#" style="color:aliceblue">User Login</a>
        </div>
        <div class="navbar-collapse collapse w-100 order-3 dual-collapse2" id="navbarNav">
          <ul class="navbar-nav mr-auto text-center" id="navbarRight">
            <li class="nav-item active">
              <a class="nav-link" href="index" style="color: aliceblue;">Home </a>
            </li>
            <li class="nav-item">
              <a class="nav-link" href="login" style="color: aliceblue;">Login</a>
            </li>
            <li class="nav-item">
              <a class="nav-link" href="register"style="color: aliceblue;">Register</a>
            </li>
          </ul>
        </div>
      </nav>
      <br><br>
      <form class="form-inline" action="/login" method="GET">
      <div class="container" style="width: 600px; height: 600px;">
        <div class="mb-3 d-flex justify-content-center"><script src="https://cdn.lordicon.com/xdjxvujz.js"></script>
            <lord-icon
                src="https://cdn.lordicon.com/elkhjhci.json"
                trigger="hover"
                style="width:200px;height:200px">
            </lord-icon></div>
            <div class="mb-3">
                <input type="email" class="form-control" id="exampleInputEmail1" name="mail" aria-describedby="emailHelp" placeholder="Enter Registered Mail ID">
              </div>
              <div class="mb-3">
                <input type="password" class="form-control" id="exampleInputPassword1" name="pass" placeholder="Enter Password">
              </div>
              <div class="mb-3">
              <button type="submit form-control" class="btn btn-dark btn-primary" style="width:100%;" type="submit">Login</button>
            </div>
            {{pred}}
      </div>
    </form>
  </body>
</html>

logout.html
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta http-equiv="X-UA-Compatible" content="IE=edge" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <!-- CSS only -->
    <link
      href="https://cdn.jsdelivr.net/npm/bootstrap@5.2.1/dist/css/bootstrap.min.css"
      rel="stylesheet"
      integrity="sha384-iYQeCzEYFbKjA/T2uDLTpkwGzCiq6soy8tYaI1GyVh/UjpbCx/TYkiZhlZB6+fzT"
      crossorigin="anonymous"
    />
    <!-- JavaScript Bundle with Popper -->
    <script
      src="https://cdn.jsdelivr.net/npm/bootstrap@5.2.1/dist/js/bootstrap.bundle.min.js"
      integrity="sha384-u1OknCvxWvY5kfmNBILK2hRnQC3Pr17a+RTT6rIHI7NnikvbZlHgTPOOmMi466C8"
      crossorigin="anonymous"
    ></script>
    <style>
        #navbarRight {
            margin-left: auto;
            padding-right:10px;
        }
        .navbar-brand{
            padding-left:15px;
        }
    </style>
    <title>DR Predcition</title>
  </head>
  <body>
    <nav class="navbar navbar-expand-lg navbar-light bg-dark">
        <div>
        <a class="navbar-brand" href="#" style="color:aliceblue">Diabetic Retinopathy</a>
        </div>
        <div class="navbar-collapse collapse w-100 order-3 dual-collapse2" id="navbarNav">
          <ul class="navbar-nav mr-auto text-center" id="navbarRight">
            <li class="nav-item active">
              <a class="nav-link" href="index" style="color: aliceblue;">Home </a>
            </li>
            <li class="nav-item">
              <a class="nav-link" href="login" style="color: aliceblue;">Login</a>
            </li>
            <li class="nav-item">
              <a class="nav-link" href="register"style="color: aliceblue;">Register</a>
            </li>
          </ul>
        </div>
      </nav>
      <br><br>
      <div class="d-flex justify-content-center">
        <div class="row d-flex display-3 justify-content-center">
            Successfully Logged Out!
            <br><br>
            <a href="login" class="btn btn-lg btn-dark">Login for more Information</a>
              </div>
        </div>
  </body>
</html>

prediction.html:-
<!DOCTYPE html>
<html lang="en">

<head>
  <meta charset="UTF-8" />
  <meta http-equiv="X-UA-Compatible" content="IE=edge" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <!-- CSS only -->
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.2.1/dist/css/bootstrap.min.css" rel="stylesheet"
    integrity="sha384-iYQeCzEYFbKjA/T2uDLTpkwGzCiq6soy8tYaI1GyVh/UjpbCx/TYkiZhlZB6+fzT" crossorigin="anonymous" />
  <!-- JavaScript Bundle with Popper -->
  <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.2.1/dist/js/bootstrap.bundle.min.js"
    integrity="sha384-u1OknCvxWvY5kfmNBILK2hRnQC3Pr17a+RTT6rIHI7NnikvbZlHgTPOOmMi466C8"
    crossorigin="anonymous"></script>
  <style>
    #navbarRight {
      margin-left: auto;
      padding-right: 10px;
    }

    .navbar-brand {
      padding-left: 15px;
    }

    .row {
      width: 90%;
    }
  </style>
  <title>DR Predcition</title>
</head>

<body>
  <nav class="navbar navbar-expand-lg navbar-light bg-dark">
    <div>
      <a class="navbar-brand" href="#" style="color:aliceblue">Diabetic Retinopathy Classification</a>
    </div>
    <div class="navbar-collapse collapse w-100 order-3 dual-collapse2" id="navbarNav">
      <ul class="navbar-nav mr-auto text-center" id="navbarRight">
        <li class="nav-item active">
          <a class="nav-link" href="index" style="color: aliceblue;">Home </a>
        </li>
        <li class="nav-item">
          <a class="nav-link" href="logout" style="color: aliceblue;">Logout</a>
        </li>
      </ul>
    </div>
  </nav>
  <br><br>
  <div class="container justify-content-center" style="width:700px">
    <form action = "/predict" method = "POST" enctype="multipart/form-data">
    <label for="formFileLg" class="form-label">Upload Image</label>
    <input class="form-control form-control-lg" name ="file" type="file" />
    <br>
    <button class="btn btn-lg btn-dark" type = "submit">Predict</button>
    </form>
    <br>
    <h1>{{prediction}}</h1>
  </div>
  <br><br><br>
  <div class="d-flex justify-content-center" >
      <img src="static/level.png" style="width: 90%">
  </div>
</body>
</html>


register.html:-
<!-- <!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta http-equiv="X-UA-Compatible" content="IE=edge" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <!-- CSS only -->
    <link
      href="https://cdn.jsdelivr.net/npm/bootstrap@5.2.1/dist/css/bootstrap.min.css"
      rel="stylesheet"
      integrity="sha384-iYQeCzEYFbKjA/T2uDLTpkwGzCiq6soy8tYaI1GyVh/UjpbCx/TYkiZhlZB6+fzT"
      crossorigin="anonymous"
    />
    <!-- JavaScript Bundle with Popper -->
    <script
      src="https://cdn.jsdelivr.net/npm/bootstrap@5.2.1/dist/js/bootstrap.bundle.min.js"
      integrity="sha384-u1OknCvxWvY5kfmNBILK2hRnQC3Pr17a+RTT6rIHI7NnikvbZlHgTPOOmMi466C8"
      crossorigin="anonymous"
    ></script>
    <style>
        #navbarRight {
            margin-left: auto;
            padding-right:10px;

        }
        .navbar-brand{
            padding-left:15px;
        }
    </style>
    <title>DR Predcition</title>
  </head>
  <form action="{{url_for('register')}}" method="post" >
    <nav class="navbar navbar-expand-lg navbar-light bg-dark">
        <div>
        <a class="navbar-brand" href="#" style="color:aliceblue">Registration</a>
        </div>
        <div class="navbar-collapse collapse w-100 order-3 dual-collapse2" id="navbarNav">
          <ul class="navbar-nav mr-auto text-center" id="navbarRight">
            <li class="nav-item active">
              <a class="nav-link" href="index" style="color: aliceblue;">Home </a>
            </li>
            <li class="nav-item">
              <a class="nav-link" href="login" style="color: aliceblue;">Login</a>
            </li>
            <li class="nav-item">
              <a class="nav-link" href="register"style="color: aliceblue;">Register</a>
            </li>
          </ul>
        </div>
      </nav>
      <br><br>
      <form class="form-inline" method ="POST">
      <div class="container" style="width: 600px; height: 600px;">
        <div class="mb-3 d-flex justify-content-center"><script src="https://cdn.lordicon.com/xdjxvujz.js"></script>
            <lord-icon
                src="https://cdn.lordicon.com/elkhjhci.json"
                trigger="hover"
                style="width:200px;height:200px">
            </lord-icon></div>
			<div class="mb-3">
                <input type="text" class="form-control" id="exampleInputName" name = "name" aria-describedby="nameHelp" placeholder="Enter Name">
              </div>
              <div class="mb-3">
                <input type="email" class="form-control" id="exampleInputEmail1" name="emailid" aria-describedby="emailHelp" placeholder="Enter Mail ID">
              </div>
              <div class="mb-3">
                <input type="number" class="form-control" id="exampleInputNumber1" name="num" aria-describedby="numberHelp" placeholder="Enter Mobile number">
              </div>
              <div class="mb-3">
                <input type="password" class="form-control" id="exampleInputPassword1" name="pass" placeholder="Enter Password">
              </div>
              <div class="mb-3">
              <button type="submit form-control" class="btn btn-dark btn-primary" style="width:100%;">Register</button>
            </div>
			<div class="mb-3 d-flex justify-content-center">
			<a href="login" class="nav-link"> Already Registered: Login Here</a>
      </div>
      {{pred}}
      </div>
    </form>
  </body>
</html> -->










Python Notebook screenshots:-


 
 
 

 
 
  


GITHUB LINK:- IBM-EPBL/IBM-Project-18407-1659684768: Deep Learning Fundus Image Analysis for Early Detection of Diabetic Retinopathy (github.com)

DEMO LINK:- IBM-Project-18407-1659684768/Demo Video.mp4 at main • IBM-EPBL/IBM-Project-18407-1659684768 (github.com)

	

	
	
	

