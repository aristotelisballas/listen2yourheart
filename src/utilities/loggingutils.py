from pathlib import Path


def log_details(details: str, tmp_filepath: Path, filename: str):
    Path.mkdir(tmp_filepath, parents=True, exist_ok=True)
    f = open(tmp_filepath / filename, "w")
    dets = f"""---------------Details---------------
    {details}
    """
    f.write(dets)
    f.close()


def get_model_summary(model):
    stringlist = []
    model.summary(print_fn=lambda x: stringlist.append(x))
    short_model_summary = "\n".join(stringlist)
    return short_model_summary
