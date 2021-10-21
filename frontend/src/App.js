import './App.css';
import React from "react";
import CanvasDraw from "react-canvas-draw";
import Module from "./predictor.js";

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
  //   const mod =  createModuleGlue({
  //     noInitialRun: true,
  //     noExitRuntime: true
  // });

  const mod = Module().then(function(result) {
    console.log(result);
    result._predict();
  });
  
    // Module._predict();

    // pred();


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
    </div>);
  }
}

export default App;