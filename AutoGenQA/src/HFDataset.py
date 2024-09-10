
from datasets import load_dataset
import random

class HFDataset:
    def __init__(self,ds_name,streaming=True) -> None:

        if not streaming:
            ds=load_dataset(ds_name,split="train",streaming=streaming).shuffle()
        else:
            try:
                ds=load_dataset(ds_name,split="train",streaming=streaming)
            except:
                ds=load_dataset(ds_name,split="validation",streaming=streaming)
        self.loader=iter(ds)
        self.ds=ds

    def __iter__(self):
        return self

    def __next__(self):
        try:
            text=next(self.loader)["text"]
        except StopIteration:
            self.loader=iter(self.ds)
            text=next(self.loader)["text"]

        return text
