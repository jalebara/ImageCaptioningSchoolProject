from utils import NLPMetricAggregator

def test_nlp_metrics():
    good_sentences = [
        "The cat in the hat knows a lot about that",
        "Today you are you! That is truer than true! There is no one alive who is you-er than you!",
        "I have great faith in fools self confidence my friends will call it."
    ]
    bad_sentences = [
        "A dog in overalls is probably unaware of many things",
        "Tomorrow I am where the red fern grows",
        "Potato potato potato potato"
    ]

    reference_sentences = [
        [
            "The cat in the hat is over there",
            "That cat sure knows a lot about that"
        ],
        [
            "Today you are you. A truer than true statement",
            "No one alive is you other than you"
        ],
        [ 
            "Have faith in fools",
            "Good friends will call it self confidence"
        ]
    ]

    good_metrics = NLPMetricAggregator()
    bad_metrics = NLPMetricAggregator()
    good_metrics.update(good_sentences, reference_sentences)
    bad_metrics.update(bad_sentences, reference_sentences)
    bad = bad_metrics.generate_metric_summaries()
    good = good_metrics.generate_metric_summaries()

    for key in good.keys():
        print(good[key])
        assert good[key] > bad[key]
