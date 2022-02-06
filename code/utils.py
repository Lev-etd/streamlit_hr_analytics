import os
import subprocess

from asrecognition import ASREngine
from pyaspeller import YandexSpeller
import torch

from transformers import AutoModelForSequenceClassification
from transformers import BertTokenizerFast
from reportlab.pdfgen.canvas import Canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.lib.units import mm
from textwrap import wrap

# import nemo.collections.asr as nemo_asr

# --------------------------------------------------------------------------------------------------------------------
# Fix encoding for russian
fname = 'a010013l'

# faceName - view a010013l.AFM file as a plain text and look at
# row beginning with 'FontName' word (it's usually the fourth row).
# The word after 'FontName' is the faceName ('URWGothicL-Book' in this case).
faceName = 'URWGothicL-Book'

# Define new Type 1 font
cyrFace = pdfmetrics.EmbeddedType1Face(fname + '.afm', fname + '.pfb')

# Create a new encoding called 'CP1251'
cyrenc = pdfmetrics.Encoding('CP1251')

# Fill in the tuple with Unicode glyphs in accordance with cp1251 (win1251)
# encoding
cp1251 = (
    'afii10051', 'afii10052', 'quotesinglbase', 'afii10100', 'quotedblbase',
    'ellipsis', 'dagger', 'daggerdbl', 'Euro', 'perthousand', 'afii10058',
    'guilsinglleft', 'afii10059', 'afii10061', 'afii10060', 'afii10145',
    'afii10099', 'quoteleft', 'quoteright', 'quotedblleft', 'quotedblright',
    'bullet', 'endash', 'emdash', 'tilde', 'trademark', 'afii10106',
    'guilsinglright', 'afii10107', 'afii10109', 'afii10108', 'afii10193',
    'space', 'afii10062', 'afii10110', 'afii10057', 'currency', 'afii10050',
    'brokenbar', 'section', 'afii10023', 'copyright', 'afii10053',
    'guillemotleft', 'logicalnot', 'hyphen', 'registered', 'afii10056',
    'degree', 'plusminus', 'afii10055', 'afii10103', 'afii10098', 'mu1',
    'paragraph', 'periodcentered', 'afii10071', 'afii61352', 'afii10101',
    'guillemotright', 'afii10105', 'afii10054', 'afii10102', 'afii10104',
    'afii10017', 'afii10018', 'afii10019', 'afii10020', 'afii10021',
    'afii10022', 'afii10024', 'afii10025', 'afii10026', 'afii10027',
    'afii10028', 'afii10029', 'afii10030', 'afii10031', 'afii10032',
    'afii10033', 'afii10034', 'afii10035', 'afii10036', 'afii10037',
    'afii10038', 'afii10039', 'afii10040', 'afii10041', 'afii10042',
    'afii10043', 'afii10044', 'afii10045', 'afii10046', 'afii10047',
    'afii10048', 'afii10049', 'afii10065', 'afii10066', 'afii10067',
    'afii10068', 'afii10069', 'afii10070', 'afii10072', 'afii10073',
    'afii10074', 'afii10075', 'afii10076', 'afii10077', 'afii10078',
    'afii10079', 'afii10080', 'afii10081', 'afii10082', 'afii10083',
    'afii10084', 'afii10085', 'afii10086', 'afii10087', 'afii10088',
    'afii10089', 'afii10090', 'afii10091', 'afii10092', 'afii10093',
    'afii10094', 'afii10095', 'afii10096', 'afii10097'
)

# Replace glyphs from code 128 to code 256 with cp1251 values
for i in range(128, 256):
    cyrenc[i] = cp1251[i - 128]

# Register newly created encoding
pdfmetrics.registerEncoding(cyrenc)

# Register type face
pdfmetrics.registerTypeFace(cyrFace)

# Register the font with adding '1251' to its name
pdfmetrics.registerFont(pdfmetrics.Font(faceName + '1251', faceName, 'CP1251'))
# --------------------------------------------------------------------------------------------------------------------

root = os.getcwd()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

asr = ASREngine("ru", model_path="jonatasgrosman/wav2vec2-large-xlsr-53-russian", device=device,
                inference_batch_size=1)

speller = YandexSpeller(lang='ru')

tokenizer = BertTokenizerFast.from_pretrained('blanchefort/rubert-base-cased-sentiment')
model = AutoModelForSequenceClassification.from_pretrained('blanchefort/rubert-base-cased-sentiment',
                                                           return_dict=True)


def save_file(video):
    with open(os.path.join("uploaded_videos", video.name), "wb") as f:
        f.write(video.getbuffer())


def run_analysis(path_to_video, path_to_audio):
    """
    Run full analysis of video
    :param path_to_video: path to video
    :param path_to_audio: path to audio
    :return: fully processed text
    """
    cmd = f"ffmpeg -i {str(path_to_video)} -ab 160k -ac 2 -ar 16000 -vn {str(path_to_audio)}"
    subprocess.run(cmd, shell=True)

    audio_paths = [str(path_to_audio)]
    transcriptions = asr.transcribe(audio_paths)

    input_text = transcriptions[0]['transcription'].lower()
    model, example_texts, languages, punct, apply_te = torch.hub.load(repo_or_dir='snakers4/silero-models',
                                                                      model='silero_te')
    punctuated_text = apply_te(input_text, lan='ru')
    text = punctuated_text

    changes = {change['word']: change['s'][0] for change in speller.spell(punctuated_text)}

    for word, suggestion in changes.items():
        text = text.replace(word, suggestion)

    return text


def run_emotion_analysis(text):
    inputs = tokenizer(text, max_length=512, padding=True, truncation=True, return_tensors='pt')
    outputs = model(**inputs)
    predicted_logits = torch.nn.functional.softmax(outputs.logits, dim=1)
    predicted = torch.argmax(predicted_logits, dim=1).numpy()
    return predicted, predicted_logits


def generate_pdf(text, emotion_info, video_name, ind_of_video):
    canvas = Canvas(f'{root}/generated_pdfs/{video_name}.pdf', pagesize=(210 * mm, 297 * mm))
    t = canvas.beginText(13 * mm, 280 * mm)
    t.setFont(faceName + '1251', 12)
    text = text
    wrapped_text = "\n".join(wrap(text, 75))
    t.textLines(wrapped_text)
    canvas.setFont(faceName + '1251', 16)
    canvas.drawString(13 * mm, 288 * mm, f'Текстовая расшифровка видео {video_name} под номером № {ind_of_video}')
    canvas.drawString(13 * mm, 150 * mm, f'Речь в видео на {round((emotion_info[0][0] * 100).item(), 2)}% нейтральна, на {round((emotion_info[0][1] * 100).item(), 2)}% позитивна и на {round((emotion_info[0][2] * 100).item(), 2)}% негативна')
    canvas.drawText(t)
    canvas.save()

