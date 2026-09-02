"""Row-mate top-padding fairness for a packed row with a mixed fan-in column.

A section's mixed full-bundle + homogeneous-subset fan-in column must receive
the same symmetric top padding as an all-full row-mate, and the shared
inter-section trunk lane must stay fixed for every row-mate in the row.
"""

from __future__ import annotations

import pytest

from nf_metro.layout.engine import compute_layout
from nf_metro.parser.mermaid import parse_metro_mermaid
from nf_metro.parser.model import PortSide

_RIBOSEQ = """\
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
%%metro file: te_out | TSV | TE results
%%metro file: report_final | HTML | MultiQC
%%metro line: riboseq | Ribo-seq | #e6007e
%%metro line: rnaseq | Matched RNA-seq | #2db572
%%metro line: tiseq | TI-seq | #2b6cb0
%%metro line: annotation | Hybrid annotation | #f2b407

%%metro grid: preprocessing, alignment, novel_transcripts | 0,0
%%metro grid: orf_calling, psite_id, te, reporting | 0,1
%%metro x_spacing: 70
%%metro legend: br

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

        ribowaltz -->|riboseq| quantify_orf_psite
        plastid_psite -->|riboseq| plastid_wiggle
        plastid_wiggle -->|riboseq| quantify_orf_psite
        plastid_wiggle -->|riboseq| psite_counts_gene
    end

    subgraph te [Translational efficiency]
        te_prep_gene[Gene count\\nmatrix]
        te_prep_orf[ORF count\\nmatrix]
        anota2seq[anota2seq]
        deltate[DESeq2 deltaTE]
        dotseq[DOTSeq]

        te_prep_gene -->|riboseq,rnaseq| anota2seq
        te_prep_gene -->|riboseq,rnaseq| deltate
        te_prep_orf -->|riboseq,rnaseq| anota2seq
        te_prep_orf -->|riboseq,rnaseq| deltate
        te_prep_orf -->|riboseq,rnaseq| dotseq
        anota2seq -->|riboseq,rnaseq| te_out
        deltate -->|riboseq,rnaseq| te_out
        dotseq -->|riboseq,rnaseq| te_out
    end

    subgraph reporting [Reporting]
        multiqc_final[MultiQC]

        multiqc_final -->|riboseq,rnaseq| report_final
    end

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
    anota2seq -->|riboseq,rnaseq| multiqc_final
    hybrid_merge -->|annotation| star_hybrid
    hybrid_merge -->|annotation| ribotish
    hybrid_merge -->|annotation| ribotricer
    hybrid_merge -->|annotation| ribocode
"""

_TOL = 1.0


@pytest.fixture(scope="module")
def graph():
    laid_out = parse_metro_mermaid(_RIBOSEQ)
    compute_layout(laid_out, validate=False)
    return laid_out


def _top_pad(graph, section_id: str) -> float:
    section = graph.sections[section_id]
    internal = [
        st.y
        for st in graph.stations.values()
        if st.section_id == section_id and not st.is_port and not st.off_track
    ]
    return min(internal) - section.bbox_y


def _lr_port_ys(graph, section_id: str) -> set[float]:
    section = graph.sections[section_id]
    ys = set()
    for pid in section.port_ids:
        port = graph.ports.get(pid)
        st = graph.stations.get(pid)
        if port and st and port.side in (PortSide.LEFT, PortSide.RIGHT):
            ys.add(round(st.y, 1))
    return ys


def test_mixed_fan_in_column_gets_row_mate_top_padding(graph):
    orf_pad = _top_pad(graph, "orf_calling")
    te_pad = _top_pad(graph, "te")
    # orf_calling's mixed fan-in must not carry a row-pitch of dead band that
    # its all-full row-mate te does not: their top padding matches.
    assert abs(orf_pad - te_pad) < _TOL, (orf_pad, te_pad)


def test_continuation_stays_on_its_fanned_predecessors_track(graph):
    # star_hybrid -> ribocode is a direct edge; ribocode also takes annotation
    # from the shared entry port, yet must ride star_hybrid's lifted track.
    assert abs(graph.stations["star_hybrid"].y - graph.stations["ribocode"].y) < _TOL


def test_shared_row_trunk_lane_is_preserved(graph):
    # the trunk lane is shared across row-mates and must not move when one
    # section's padding is corrected.
    lanes = set()
    for sid in ("orf_calling", "psite_id", "te", "reporting"):
        lanes |= _lr_port_ys(graph, sid)
    assert len(lanes) == 1, lanes
