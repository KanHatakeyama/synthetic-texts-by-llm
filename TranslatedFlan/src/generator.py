import random


def prepare_records(ds, mode_list,
                    inst_dict,
                    n_records=300,
                    ):
    ds = ds.shuffle()

    records = []
    cnt = 0
    for record in ds:
        mode = random.choice(mode_list)
        inst = inst_dict[mode]

        text = record["inputs"]+"\n"
        text += record["targets"]
        text = f"""<|user|>
{inst}{text}<|end|>
<|assistant|>#日本語\n"""

        records.append(
            {"original_text": text,
                "mode": mode,
                "en": record["inputs"]+"\n"+record["targets"]
             }
        )
        cnt += 1
        if cnt > n_records:
            break

    return records
