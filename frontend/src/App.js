import './App.css';
import React from "react";
import CanvasDraw from "react-canvas-draw";
import Module from "./predictor.js";

class App extends React.Component {
  constructor(props) {
    super(props);
    this.handleClear = this.handleClear.bind(this);
    this.handlePredict = this.handlePredict.bind(this);
  }

  state = {
    color: "white",
    size: 420,
    lazyRadius: 12,
    probs: {...[0,0,0,0,0,0,0,0,0,0]},
    img_data: null,
    pixel_vec_flat: null,
  };
  
  handleClear(e) {
    console.log("Adsfasdfdasf");
    this.canvas.clear();
  };

  handlePredict() {
    let a = Array(0);
    this.setState({img_data: this.canvas.canvas.drawing.getContext("2d").getImageData(0, 0, this.state.size, this.state.size)}, () => {
      let pooling_size = this.state.size / 28;
      let height = this.state.img_data.height;
      let width = this.state.img_data.width;
      console.log(this.state.img_data);

      for (let i = 0; i < height; i = i + pooling_size) {
        for (let j = 0; j < width; j = j + pooling_size) {
            let sum = 0;
            for (let y = i; y < i + pooling_size; y++) {
                for (let x = j; x < j + pooling_size; x++) {
                  let idx = y * (width * 4) + x * 4;
                    sum += this.state.img_data.data[idx] / 255.0;
                }
            }
            a.push(sum / (pooling_size * pooling_size));
        }   
      }

      console.log(a);

      let b = Array(0);

      for (let z = 0; z < 5; z++) {
        b = b.concat(a);
      }

      console.log(this.state);

      this.setState({pixel_vec_flat: b}, () => {
        let vec = this.state.pixel_vec_flat;
        console.log(this.state);
        const mod = Module().then((result) => {
          var v1 = new Float64Array(vec);
          v1[0] = 300;
          var uarray = new Uint8Array(v1.buffer)

          let pred = result.cwrap('predict', 'number', ['array', 'number']);
          let predictions = Array(10);


          for (let i = 0; i < 10; i++) {
            predictions[i] = pred(uarray, i)
          }
          // predictions[0] = pred(uarray, 0)
          console.log(predictions)
          
          this.setState({probs: predictions});
          // console.log(this.state);

        });
      });
  });};

  render() {
    return(         
    <div className="App">
      <button onClick={this.handleClear}> Clear </button>
      <button onClick={this.handlePredict}> Predict </button>
      <CanvasDraw
          ref={canvasDraw => (this.canvas = canvasDraw)}
          brushColor="white"
          brushRadius={25}
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