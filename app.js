const firebaseConfig = {
    apiKey: "AIzaSyC5AcmbhTu16qKp-KBHlvX0refA95FYGRg",
    authDomain: "hackaton-ai-aab50.firebaseapp.com",
    databaseURL: "https://hackaton-ai-aab50-default-rtdb.firebaseio.com",
    projectId: "hackaton-ai-aab50",
    storageBucket: "hackaton-ai-aab50.appspot.com",
    messagingSenderId: "528769515799",
    appId: "1:528769515799:web:72c7baa7355c2b3c54f752",
    measurementId: "G-BM2ECYSSK1"
};

const app = firebase.initializeApp(firebaseConfig);

// window.onload = (event) =>{
//     set_data()
//     console.log("window onload")
//     setTimeout(function(){
//         get_profile()
//     }, 400);
// };


/**
 * Writes data into realtime database for users
 * @param dic
 */
function writeUserData(dic) {
    const dbRef = firebase.database();
    dbRef.ref("prompt/").update(dic);
}

writeUserData({"ger": "de"})

/**
 * Reads data into realtime database for users with connection
 * @param userId
 */
function readUserData(_callback) {
    const dbRef = firebase.database();
    dbRef.ref('prompt/').on('value', (snapshot) => {
        if (snapshot.exists()) {
            // return snapshot.val()
            _callback(snapshot.val());
        } else {
            console.log("No data available");
        }
    }).then(() => {
        console.log("verification email sent!")
    });
}

readUserData((val)=>{
    console.log(val)
})