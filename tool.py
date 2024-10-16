from flask import Flask, render_template, request, make_response, jsonify, Response, send_file
from helper_functions import *
import time
from datetime import datetime
import io
from pathlib import Path

app = Flask(__name__)
@app.route('/download_cm', methods=["POST"])
def download_cm():
    if request.method == "POST":
        p = Path("CMs")
        p.mkdir(exist_ok=True)

        now = datetime.now()
        current_time = now.strftime("%H-%M-%S")

        req = request.get_json()
        # print(req)
        actions_set_form = req['actions']
        gestures_set_form = req['gestures']
        # print(gestures_set_form)
        imus_set_form = req['imus']
        imu_count_form = int(req['imu_count'])

        if ((len(actions_set_form) < 1) | (len(gestures_set_form) < 2) | (len(imus_set_form) < 1)):
            # can't train a model
            response = {
                    'message': 'Something is missing',
                    }
            return make_response(jsonify(response), 200)

        else:
            # train a model and return feature importances
            print("Training")
            start = time.time()
            return_val = train_subset(X, y, actions_set_form, gestures_set_form, imus_set_form, imu_count_form)
            cm_df = return_val["cm_df"]
            end = time.time()
            print(f"Done Training, elapsed time {end-start}")

            cm_df.to_csv(p / f"{current_time}.csv")
            # best_imus = df_imu_fi.nlargest(imu_count_form, columns=["Feature importance"], keep="all").index.to_list()
            # print(best_imus)

            response = {
                    'message': 'Message received, your CM should be downloaded in CMs'
                    }

            return make_response(jsonify(response), 200)

@app.route('/get_cm', methods=["POST"])
def get_cm():
    if request.method == 'POST':
        req = request.get_json()
        # print(req)
        actions_set_form = req['actions']
        gestures_set_form = req['gestures']
        # print(gestures_set_form)
        imus_set_form = req['imus']
        imu_count_form = int(req['imu_count'])

        if ((len(actions_set_form) < 1) | (len(gestures_set_form) < 2) | (len(imus_set_form) < 1)):
            # can't train a model
            response = {
                    'message': 'Something is missing',
                    }
            return make_response(jsonify(response), 200)

        else:
            # train a model and return feature importances
            print("Training")
            start = time.time()
            return_val = train_subset(X, y, actions_set_form, gestures_set_form, imus_set_form, imu_count_form)
            cm_image = return_val["image"]
            accuracy = return_val["accuracy"]
            best_imus = return_val['best_imus']
            f1 = return_val['f1']
            end = time.time()
            print(f"Done Training, elapsed time {end-start}")

            # best_imus = df_imu_fi.nlargest(imu_count_form, columns=["Feature importance"], keep="all").index.to_list()
            # print(best_imus)

            response = {
                    'message': 'Message received, here is your CM!',
                    'cm': cm_image,
                    'accuracy': accuracy,
                    'best_imus': best_imus,
                    'f1': f1
                    }

            return make_response(jsonify(response), 200)

@app.route('/get_gestures', methods=['POST'])
def get_gestures():
    if request.method == 'POST':
        # get a set of actions everytime a checkbox value changes
        req = request.get_json()
        # actions_set_form = [f" {action}" for action in req['actions']]
        actions_set_form = req['actions']
        gestures_set = get_finger_gestures_set(X, y, actions_set_form)

        # remove unscripted action and activity
        finger_gestures_set = [gesture for gesture in gestures_set if not (("Unscripted" in gesture) or ("Activity" in gesture))]

        response = {
                'message': 'Actions Received, Here are your gestures',
                'finger_gestures': finger_gestures_set,
                }

        return make_response(jsonify(response), 200)

@app.route('/get_feature_importances', methods=['POST'])
def get_feature_importances():
    if request.method == 'POST':
        req = request.get_json()
        # print(req)
        actions_set_form = req['actions']
        gestures_set_form = req['gestures']
        imus_set_form = req['imus']

        if ((len(actions_set_form) < 1) | (len(gestures_set_form) < 2) | (len(imus_set_form) < 1)):
            # can't train a model
            response = {
                    'message': 'Something is missing',
                    }
            return make_response(jsonify(response), 200)

        else:
            # train a model and return feature importances
            print("Training")
            df_imu_fi = train_subset(X, y, actions_set_form, gestures_set_form, imus_set_form)
            print("Done Training")

            response = {
                    'message': 'Message received, here are your feature importances!',
                    'labels': df_imu_fi.index.to_list(),
                    'fi': list(df_imu_fi["Feature importance"].values)
                    }

            return make_response(jsonify(response), 200)

@app.route('/')
def root():
    return render_template('home.html', html_data=html_data)

if __name__ == '__main__':
    path = 'processed_dataset/'
    X, y = read_data(path)
    actions = y['Action'].unique()

    actions_with_grasps = {
            "Cylindrical": {
                "id": [" Cutting_Knife", " Drinking_Beer"],
                "value": ["Small (eg: Knife)", "Large (eg: Bottle)"]
                },
            "Palmer": {
                "id": [' Reading_Book', ' Carrying_Box' ],
                "value": ["Small (eg: Book)", "Large (eg: Box)"]
                },
            "Hook": {
                "id": [' Holding_Cup', ' Carrying_Bag' ],
                "value": ["Small (eg: Cup)", "Large (eg: Bag)"]
                },
            "Lateral": {
                "id": [' Pouring_Spoon' , ' Reading_paper'],
                "value": ["Small (eg: Spoon)", "Large (eg: Paper)"]
                },
            "Tip": {
                "id": [' Needle_Sewing', ' Writing_Pen'],
                "value": ["Small (eg: Needle)", "Large (eg: Pen)"]
                },
            "Spherical": {
                "id": [' Crushing_Pestle', ' Bowl_Placing'],
                "value": ["Small (eg: Pestle)", "Large (eg: Bowl)"]
                },
            "FreeHand": {
                "id": [' FreeHand'],
                "value": ["FreeHand"]
                }
            }

    gestures = y['Class_true_w_finger'].unique()

    imus = ["F1_prox", "F1_midd", "F1_dist", 
            "F2_prox", "F2_midd", "F2_dist",
            "F3_prox", "F3_midd", "F3_dist",
            "F4_prox", "F4_midd", "F4_dist",
            "F5_prox", "F5_midd", "F5_dist",
            "handback", "forearm"]
    
    joints = ["prox", "midd", "dist"]
    joint_names = ["Proximal", "Middle", "Distal"]
    imu_fingers = ["F1", "F2", "F3", "F4", "F5", "handback", "forearm"]
    imu_fingers_names = {
            "F1": "Thumb",
            "F2": "Index",
            "F3": "Middle",
            "F4": "Ring",
            "F5": "Pinky",
            "handback": "Handback",
            "forearm": "Wrist"
            }

    # remove Unscripted Action and Activity
    gestures = sorted(list(set([gesture.split("_")[0] for gesture in gestures if not (("Unscripted" in gesture) or ("Activity" in gesture))])))

    fingers = ["Thumb", "Index", "Middle"]

    html_data = {
            'Actions': actions,
            'Action_w_grasps': actions_with_grasps,
            'Gestures': gestures,
            'Fingers': fingers,
            'IMUs': imus,
            'joints': joints,
            'joint_names': joint_names,
            'imu_fingers': imu_fingers,
            'imu_fingers_names': imu_fingers_names
            }

    app.run(host='localhost', port=4444, debug=True)
