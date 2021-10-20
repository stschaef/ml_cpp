import './App.css';
import React from "react";
import CanvasDraw from "react-canvas-draw";

class App extends React.Component {
  constructor(props) {
    super(props);
    this.handleClear = this.handleClear.bind(this);
  }

  state = {
    color: "white",
    size: 400,
    lazyRadius: 12,
      probs: {...[0,0,0,0,0,0,0,0,0,0]}
  };
  
  handleClear(e) {
    this.canvas.clear();
  };

  handlePredict(e) {
    console.log("Adsf");
  };

  render() {
    var pred = Module.cwrap('predict', null, [Number]);

    pred();


    return(         
    <div className="App">
      <button onClick={this.handleClear}> Clear </button>
      <button onClick={this.handlePredict}> Predict </button>
      <CanvasDraw
          ref={canvasDraw => (this.canvas = canvasDraw)}
          brushColor="white"
          brushRadius={10}
          canvasWidth={this.state.size}
          canvasHeight={this.state.size}
          backgroundColor="black"
          hideGrid="true"
          lazyRadius={0}
        />
    <ul>
      {Object.keys(this.state.probs).map((num) => (
        <li key={num}> {num} - Prob: {this.state.probs[num]}  </li>
      ))}
    </ul>
    <link rel="import" href="../../predictor.html"/>
    <predictor/>
    </div>);
  }
}

export default App;