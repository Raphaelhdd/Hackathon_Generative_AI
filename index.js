// START - HANDLE FORMS

function handleForms () {

document.getElementById("see-my-project").addEventListener("click", (e)=>{
    titleIdea = document.getElementById("title_input").value;
    ideaDescription = document.getElementById("area-test").value;
    attributes = []
    for (i = 0; i < document.querySelectorAll(".form-group .row").length; i++) {
        attributes.push(document.getElementsByName("kt_docs_repeater_basic[" + i + "][account_name]")[0].value)
    }

    writeUserData({'title': titleIdea, 'ideaDescription': ideaDescription, 'attributes': attributes}, ()=>{
        setTimeout(()=>{
            window.location.href = "steps.html"
        }, 300)

    })
})
}

// END - HANDLE FORMS







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

function write_known_key(newTeamKey, dic, _callback) {
    const dbRef = firebase.database();
    dbRef.ref("prompt/" + newTeamKey + "/").update(dic).then(() => {
        _callback()
    });
}
/**
 * Writes data into realtime database for users
 * @param dic
 */
function writeUserData(dic, _callback) {
    var newTeamKey = firebase.database().ref().child('matches').push().key;
    localStorage.setItem("key", newTeamKey);
    write_known_key(newTeamKey, dic, ()=>{
      _callback()
    })
}


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
    });

}

readUserData((val)=>{
    console.log(val)
})
function init() {
    var RANDOM_TEXT = "IDEA AND RESEARCH"

    // Since 2.2 you can also author concise templates with method chaining instead of GraphObject.make
    // For details, see https://gojs.net/latest/intro/buildingObjects.html
    const $ = go.GraphObject.make;  // for conciseness in defining templates
    var baseColor = tinycolor("#95c2f5");

// Generate a palette of 10 colors
    var blues = ["#95c2f5"];
    for (var i = 0; i < 20; i++) {
        var modifiedColor = baseColor.saturate(10).lighten(5);
        blues.push(modifiedColor.toHexString());
    }
    // var blues = [
    //     'transparent',
    //     'transparent'];

    myDiagram = $(go.Diagram, "myDiagramDiv",  // must name or refer to the DIV HTML element
            {
                initialContentAlignment: go.Spot.Center,
                layout:
                    $(go.TreeLayout,
                        {
                            treeStyle: go.TreeLayout.StyleLastParents,
                            arrangement: go.TreeLayout.ArrangementHorizontal,
                            // properties for most of the tree:
                            angle: 90,
                            layerSpacing: 35,
                            // properties for the "last parents":
                            alternateAngle: 90,
                            alternateLayerSpacing: 35,
                            alternateAlignment: go.TreeLayout.AlignmentBus,
                            alternateNodeSpacing: 20
                        }),
                // moving and copying nodes also moves and copies their subtrees
                "commandHandler.copiesTree": true,  // for the copy command
                "commandHandler.deletesTree": true, // for the delete command
                "draggingTool.dragsTree": true,  // dragging for both move and copy
                "undoManager.isEnabled": true
            });
    myDiagram.contentAlignment = go.Spot.Center,




    // Define the Node template.
    // This uses a Spot Panel to position a button relative
    // to the ellipse surrounding the text.
    myDiagram.nodeTemplate =
        $(go.Node, "Spot",
            {
                selectionObjectName: "PANEL",
                isTreeExpanded: false,
                isTreeLeaf: false
            },
            // the node's outer shape, which will surround the text
            $(go.Panel, "Auto",
                {name: "PANEL"},
                $(go.Shape, "RoundedRectangle",
                    {fill: "transparent", stroke: "#c1cee8", "strokeWidth": 5,},
                    new go.Binding("fill", "rootdistance", dist => {
                        dist = Math.min(blues.length - 1, dist);
                        return blues[dist];
                    }
                    )
                ),
                $(go.TextBlock,
                    {
                        font: "15pt sans-serif",
                        margin: 15,
                        editable: true,
                        width: 250
                    },
                    new go.Binding("text"))
            ),

            // the expand/collapse button, at the top-right corner
            $("TreeExpanderButton",
                {
                    "ButtonBorder.figure": "Circle",
                    "ButtonBorder.fill": "#b6dbff",
                    "ButtonBorder.stroke": "#87bcf5",
                    "ButtonBorder.strokeWidth": 3,
                    "_buttonFillOver": "#87bcf5",
                    "_buttonStrokeOver": "#87bcf5",
                    "_buttonFillPressed": "#87bcf5",

                    name: 'TREEBUTTON',
                    width: 30, height: 30,
                    alignment: go.Spot.TopRight,
                    alignmentFocus: go.Spot.Center,
                    // customize the expander behavior to
                    // create children if the node has never been expanded
                    click: (e, obj) => {  // OBJ is the Button
                        var node = obj.part;  // get the Node containing this Button
                        if (node === null) return;
                        e.handled = true;
                        expandNode(node);
                    }
                }
            )  // end TreeExpanderButton
        );  // end Node

    // create the model with a root node data
    myDiagram.model = new go.TreeModel([
        {text:  RANDOM_TEXT,
            color: blues[0],
            everExpanded: false}
    ]);



}
flag = false
function expandNode(node) {
    var diagram = node.diagram;
    diagram.startTransaction("CollapseExpandTree");
    // this behavior is specific to this incrementalTree sample:
    var data = node.data;
    if (flag) {
        write_known_key( localStorage.getItem("key"), {"click": data.text}, () => {
            var databaseRef = firebase.database().ref('promptOutput/' + localStorage.getItem("key"));
            // Set up event listener
            databaseRef.on('value', function (snapshot) {
                // This callback function will be called whenever there are changes in the database
                var data = snapshot.val();
                console.log(data)
                numchildren = data.response.length
                result_test = data.response
            });
        })
    }
    flag = true
    if (!data.everExpanded) {
        // only create children once per node
        diagram.model.setDataProperty(data, "everExpanded", true);
        var numchildren = createSubTree(data);
        if (numchildren === 0) {  // now known no children: don't need Button!
            node.findObject('TREEBUTTON').visible = false;
        }
    }
    // this behavior is generic for most expand/collapse tree buttons:
    if (node.isTreeExpanded) {
        diagram.commandHandler.collapseTree(node);
    } else {
        diagram.commandHandler.expandTree(node);
    }
    diagram.commitTransaction("CollapseExpandTree");
    // myDiagram.zoomToFit();
}
var numchildren = 3
// This dynamically creates the immediate children for a node.
// The sample assumes that we have no idea of whether there are any children
// for a node until we look for them the first time, which happens
// upon the first tree-expand of a node.
function createSubTree(parentdata) {

    // create several node data objects and add them to the model
    var model = myDiagram.model;
    var parent = myDiagram.findNodeForData(parentdata);

    var degrees = 1;
    var grandparent = parent.findTreeParentNode();
    while (grandparent) {
        degrees++;
        grandparent = grandparent.findTreeParentNode();
    }
    var child;
    for (var i = 0; i < numchildren; i++) {
            var childdata = {
                text: result_test[i],
                key: model.nodeDataArray.length,
                parent: parentdata.key,
                rootdistance: degrees
            };
        // add to model.nodeDataArray and create a Node
        model.addNodeData(childdata);

        // position the new child node close to the parent
        child = myDiagram.findNodeForData(childdata);

        child.location = parent.location;

    };
    return numchildren;
}
var result_test = []
function expandAtRandom() {
    var eligibleNodes = [];
    myDiagram.nodes.each(n => {
        if (!n.isTreeExpanded) eligibleNodes.push(n);
    });
    var node = eligibleNodes[Math.floor(Math.random() * (eligibleNodes.length))];
    expandNode(node);
}
if (window.location.href.includes("diagram.html")) {
    window.addEventListener('DOMContentLoaded', init);
    var databaseRef = firebase.database().ref('promptOutput/' + localStorage.getItem("key"));
    // Set up event listener
    databaseRef.on('value', function(snapshot) {
        // This callback function will be called whenever there are changes in the database
        var data = snapshot.val();
        console.log(data)
        numchildren = data.response.length
        result_test = data.response
    });
}
else {
    handleForms()
}