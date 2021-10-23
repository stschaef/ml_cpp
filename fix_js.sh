#!/usr/bin/env bash

sed -i "1s/.*/\/* eslint-disable *\//" predictor.js

f1="var _scriptDir = import.meta.url"
r1="var _scriptDir = '/"
sed -i "s/$f1/$r1/" predictor.js

f2="self.location.href"
r2="window.self.location.href"
sed -i "s/$f2/$r2/" predictor.js


