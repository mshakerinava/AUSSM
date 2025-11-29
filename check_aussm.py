import sys
import os

# Add extension-cpp to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'extension-cpp'))

# Now import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'ai-modules'))
from wavesAI.model.aussm import SSMSeq2Seq 
# exit()

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'ai-modules'))
from wavesAI.model.aussm import SSMSeq2Seq

model = SSMSeq2Seq(
    d_model=64,
    vocab_size=1000,
    output_vocab_size=1000,
    layers="m|m|a"
)

print("successfully loaded the model") 