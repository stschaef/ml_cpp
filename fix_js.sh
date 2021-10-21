#!/usr/bin/env bash

sed -i "1s/.*/\/* eslint-disable *\//" predictor.js

f1="var _scriptDir = import.meta.url"
r1="var _scriptDir = '..\/public\/predictor.wasm'"
sed -i "s/$f1/$r1/" predictor.js

f2="self.location.href"
r2="window.self.location.href"
sed -i "s/$f2/$r2/" predictor.js

sed -i "460,464 s/^/\/\//" predictor.js

sed -i "468,475 s/^/\/\//" predictor.js

r3="const wasmBinaryFile = '..\/public\/predictor.wasm'"
sed -i "466 s/.*/$r3/" predictor.js

sed -i "477,490 s/^/\/\//" predictor.js

