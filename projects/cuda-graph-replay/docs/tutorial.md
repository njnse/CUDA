# Replay, fuse, or keep ordinary launches?

Start with the eager path. Two launches represent a small pipeline whose operations depend on one another. The stream establishes the affine-to-ReLU dependency.

Capture those same launches between cudaStreamBeginCapture and cudaStreamEndCapture. Instantiate the resulting graph and upload it once. cudaGraphLaunch then submits the captured work. Buffers and parameters remain alive and unchanged for all replays. A two-node assertion catches accidental capture changes.

The fused comparator performs both operations in one kernel, removing the temporary-array traffic and one launch. If a real application can do this, fusion may be the simpler solution. Graph replay becomes more interesting when a realistic pipeline cannot be fused but executes many times with stable dependencies.

Run correctness before timing. Inspect all three timing scopes. A low device interval with a higher host wall interval can indicate submission or synchronization overhead. Do not present host submission time as GPU execution time.

Finally amortize setup. Graph construction is useful only when repeated savings exceed setup and maintenance cost. Larger arrays can make memory traffic dominate, reducing the relative importance of launch overhead. Actual device and driver measurements decide the result; no universal winner is assumed.
