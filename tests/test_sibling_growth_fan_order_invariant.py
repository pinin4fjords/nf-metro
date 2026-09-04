"""A sibling section's off-track growth must not reorder an unrelated fan.

Regression lock for #1929.  ``orf_calling`` and ``psite_id`` share one
authored grid row.  Adding a single edge inside ``psite_id`` that routes a
line across a new file sink legitimately off-tracks that sink and makes
``psite_id`` taller.  Stage 4.7 then top-aligns the whole row, growing
``orf_calling``'s bbox above its content.  Stage 6.11's fan-balance pass
must not read that bbox growth as room to lift a below-trunk sibling:
``orf_calling``'s content is unchanged, so its five-way fan order and its
reconvergence's position on the entry-port centreline must not move.

The two fixtures differ by exactly one edge line inside ``psite_id``.  The
invariant: ``orf_calling``'s internal station order is identical across the
two, and its reconvergence stays at the vertical centre of the fan.
"""

from __future__ import annotations

import warnings

import pytest
from conftest import parse_and_layout

from nf_metro.api import prepare_graph
from nf_metro.layout.constants import SAME_COORD_TOLERANCE
from nf_metro.layout.phases._common import _section_lr_port_anchor_y

_BASE = """\
%%metro title: nf-core/riboseq
%%metro center_ports: true
%%metro style: dark
%%metro diamond_style: symmetric
%%metro directional: true
%%metro file: fastq_in | FASTQ
%%metro file: hybrid_gtf_out | GTF | Hybrid GTF
%%metro file: orf_catalogue | BED | ORF catalogue
%%metro file: bigwig_out | BW | Coverage
%%metro file: counts_out | TSV | Gene counts
%%metro file: psite_orf_out | TSV | ORF P-site counts
%%metro file: te_out | TSV | TE results
%%metro file: report_final | HTML | MultiQC
%%metro line: riboseq | Ribo-seq | #e6007e
%%metro line: rnaseq | Matched RNA-seq | #2db572
%%metro line: tiseq | TI-seq | #2b6cb0
%%metro line: annotation | Hybrid annotation | #f2b407

%%metro grid: preprocessing, alignment, novel_transcripts | 0,0
%%metro grid: orf_calling, psite_id, te, reporting | 0,1
%%metro x_spacing: 70

graph LR
    subgraph preprocessing [Read pre-processing]
        fastq_in[ ]
        umi_extract[UMI-tools extract]
        fastp[fastp]
        trimgalore[Trim Galore!]
        bbsplit[BBSplit]
        sortmerna[SortMeRNA]
        ribodetector[RiboDetector]
        bowtie2_rrna[Bowtie2]
        fastqc[FastQC]
        infer_strand[Infer strandedness]
        equalise[Equalise\\nread lengths]

        fastq_in -->|riboseq,rnaseq,tiseq| umi_extract
        umi_extract -->|riboseq,rnaseq,tiseq| fastp
        umi_extract -->|riboseq,rnaseq,tiseq| trimgalore
        fastp -->|riboseq,rnaseq,tiseq| bbsplit
        trimgalore -->|riboseq,rnaseq,tiseq| bbsplit
        bbsplit -->|riboseq,rnaseq,tiseq| sortmerna
        bbsplit -->|riboseq,rnaseq,tiseq| ribodetector
        bbsplit -->|riboseq,rnaseq,tiseq| bowtie2_rrna
        sortmerna -->|riboseq,rnaseq,tiseq| fastqc
        ribodetector -->|riboseq,rnaseq,tiseq| fastqc
        bowtie2_rrna -->|riboseq,rnaseq,tiseq| fastqc
        fastqc -->|riboseq,rnaseq,tiseq| infer_strand
        infer_strand -->|riboseq,rnaseq,tiseq| equalise
    end

    subgraph alignment [Alignment & quantification]
        star[STAR]
        umi_dedup[UMI-tools dedup]
        genomecov[BEDTools\\ngenomecov]
        salmon_quant[Salmon]
        bigwig_out[ ]
        counts_out[ ]

        star -->|riboseq,rnaseq,tiseq| umi_dedup
        umi_dedup -->|riboseq,rnaseq,tiseq| genomecov
        genomecov -->|riboseq,rnaseq,tiseq| bigwig_out
        umi_dedup -->|riboseq,rnaseq,tiseq| salmon_quant
        salmon_quant -->|riboseq,rnaseq,tiseq| counts_out
    end

    subgraph novel_transcripts [Transcript discovery]
        stringtie[StringTie]
        gffcompare[gffcompare]
        hybrid_merge[Merge &\\nfilter GTF]
        hybrid_gtf_out[ ]

        stringtie -->|rnaseq| gffcompare
        gffcompare -->|rnaseq| hybrid_merge
        hybrid_merge -->|rnaseq| hybrid_gtf_out
    end


    subgraph orf_calling [ORF discovery & calling]
        star_hybrid[STAR:\\nhybrid 2nd pass]
        ribotish[Ribo-TISH]
        ribocode[RiboCode]
        ribotricer[Ribotricer]
        rpbp[Rp-Bp]
        price[PRICE]
        orf_merge[Merge ORF\\ncatalogue]
        orf_catalogue[ ]

        star_hybrid -->|riboseq| ribocode
        ribotish -->|riboseq| orf_merge
        ribocode -->|riboseq| orf_merge
        ribotricer -->|riboseq| orf_merge
        rpbp -->|riboseq| orf_merge
        price -->|riboseq| orf_merge
        orf_merge -->|riboseq| orf_catalogue
    end

    subgraph psite_id [P-site identification]
        ribowaltz[riboWaltz]
        plastid_psite[plastid\\nP-site]
        plastid_wiggle[plastid\\nwiggle]
        quantify_orf_psite[Quantify ORF\\nP-sites]
        psite_counts_gene[Gene in-frame\\nP-sites]
        psite_orf_out[ ]

        ribowaltz -->|riboseq| quantify_orf_psite
        plastid_psite -->|riboseq| plastid_wiggle
        plastid_wiggle -->|riboseq| quantify_orf_psite
        plastid_wiggle -->|riboseq| psite_counts_gene
@EXTRA_EDGE@    end

    subgraph te [Translational efficiency]
        te_prep_gene[Gene count\\nmatrix]
        te_prep_orf[ORF count\\nmatrix]
        anota2seq[anota2seq]
        deltate[DESeq2 deltaTE]
        dotseq[DOTSeq]
        _te_merge[ ]
        te_out[ ]

        te_prep_gene -->|riboseq,rnaseq| anota2seq
        te_prep_gene -->|riboseq,rnaseq| deltate
        te_prep_orf -->|riboseq,rnaseq| anota2seq
        te_prep_orf -->|riboseq,rnaseq| deltate
        te_prep_orf -->|riboseq,rnaseq| dotseq
        anota2seq -->|riboseq,rnaseq| _te_merge
        deltate -->|riboseq,rnaseq| _te_merge
        dotseq -->|riboseq,rnaseq| _te_merge
        _te_merge -->|riboseq,rnaseq| te_out
    end

    subgraph reporting [Reporting]
        multiqc_final[MultiQC]
        report_final[ ]

        multiqc_final -->|riboseq,rnaseq| report_final
    end

    %% Inter-section edges
    equalise -->|riboseq,rnaseq,tiseq| star
    equalise -->|riboseq| star_hybrid
    umi_dedup -->|rnaseq| stringtie
    umi_dedup -->|riboseq| ribotish
    umi_dedup -->|riboseq| ribotricer
    umi_dedup -->|riboseq| rpbp
    umi_dedup -->|riboseq| price
    umi_dedup -->|riboseq| ribowaltz
    umi_dedup -->|riboseq| plastid_psite
    orf_merge -->|riboseq| quantify_orf_psite
    salmon_quant -->|rnaseq| te_prep_gene
    salmon_quant -->|rnaseq| te_prep_orf
    psite_counts_gene -->|riboseq| te_prep_gene
    quantify_orf_psite -->|riboseq| te_prep_orf
    _te_merge -->|riboseq,rnaseq| multiqc_final
    hybrid_merge -->|annotation| star_hybrid
    hybrid_merge -->|annotation| ribotish
    hybrid_merge -->|annotation| ribotricer
    hybrid_merge -->|annotation| ribocode
"""

# The only difference: one extra edge routing ``riboseq`` across the new
# ``psite_orf_out`` file sink inside the sibling ``psite_id`` section, which
# legitimately off-tracks that sink and grows the shared grid row.
_EXTRA_EDGE = "        quantify_orf_psite -->|riboseq| psite_orf_out\n"

WITHOUT_SINK = _BASE.replace("@EXTRA_EDGE@", "")
WITH_SINK = _BASE.replace("@EXTRA_EDGE@", _EXTRA_EDGE)

# The five-way fan-out column of ``orf_calling`` and its reconvergence.
_FAN_COLUMN = ("star_hybrid", "ribotish", "ribotricer", "rpbp", "price")
_RECONVERGENCE = "orf_merge"


def _orf_calling_internal_order(text: str) -> list[str]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        graph = prepare_graph(text)
    section = graph.sections["orf_calling"]
    rows = sorted(
        (round(graph.stations[sid].y, 1), round(graph.stations[sid].x, 1), sid)
        for sid in section.station_ids
        if not graph.stations[sid].is_port
        and not graph.stations[sid].is_hidden
        and not graph.stations[sid].off_track
    )
    return [sid for _, _, sid in rows]


def _fan_is_trunk_centred(text: str) -> tuple[float, float]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        graph = prepare_graph(text)
    fan_ys = [graph.stations[sid].y for sid in _FAN_COLUMN]
    fan_mid = (min(fan_ys) + max(fan_ys)) / 2
    return fan_mid, graph.stations[_RECONVERGENCE].y


def test_sibling_off_track_growth_preserves_orf_calling_order() -> None:
    """The extra ``psite_id`` sink must not reorder ``orf_calling``'s fan."""
    assert _orf_calling_internal_order(WITHOUT_SINK) == _orf_calling_internal_order(
        WITH_SINK
    )


@pytest.mark.parametrize(
    "text", [WITHOUT_SINK, WITH_SINK], ids=["without_sink", "with_sink"]
)
def test_orf_calling_reconvergence_is_fan_centred(text: str) -> None:
    """``Merge ORF catalogue`` stays at the vertical centre of the fan."""
    fan_mid, reconvergence_y = _fan_is_trunk_centred(text)
    assert abs(fan_mid - reconvergence_y) <= SAME_COORD_TOLERANCE


@pytest.mark.parametrize(
    "text", [WITHOUT_SINK, WITH_SINK], ids=["without_sink", "with_sink"]
)
def test_orf_calling_trunk_stays_on_row_grid(text: str) -> None:
    """``orf_calling``'s trunk sits an integer slot count from its row siblings.

    The off-track P-site sinks grow ``psite_id`` and open top slack in the
    row-mate ``orf_calling``.  Fanning a fan-in branch into that slack would
    drag the reconvergence join a half slot off the row grid; the join must
    instead stay an exact multiple of ``y_spacing`` from the sibling trunk.
    """
    y_spacing = 55.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        graph = parse_and_layout(text, y_spacing=y_spacing)
    orf_port = _section_lr_port_anchor_y(graph, graph.sections["orf_calling"])
    sibling_port = _section_lr_port_anchor_y(graph, graph.sections["psite_id"])
    slots = (orf_port - sibling_port) / y_spacing
    assert abs(slots - round(slots)) * y_spacing <= SAME_COORD_TOLERANCE
