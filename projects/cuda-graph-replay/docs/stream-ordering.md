# Why the large-input regression mattered

A synchronous cudaMemcpy call is not a universal dependency edge between a default-stream transfer and an independently created nonblocking stream. Host staging and device completion semantics matter. Relying on incidental completion can make small inputs pass while a larger input exposes incorrect output.

This project now queues initial input and poison transfers with cudaMemcpyAsync on the same explicit execution stream as the kernels. Initial input is synchronized before capture/setup timing. Poison transfers precede the tested launch on that stream, and D2H verification follows stream synchronization. Copies remain outside replay benchmark timing.

The one-million-element case is retained as a regression test, alongside NaN-poisoned outputs, exact comparison and sanitizer checks. The correction addresses ordering, not numerical tolerance.
