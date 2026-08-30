"""Merge-fed confluence keeps a shared band and its descent on one nesting order.

On the nf-core/riboseq map the ``annotation`` and ``riboseq`` lines converge into
``orf_calling``'s LEFT entry port from different sources -- ``annotation`` around
the section as an exempt wrap, ``riboseq`` through a merge junction -- sharing one
inter-row band and one descent column.  The co-travelling separation pass reorders
the band (the next row's header blocks the downward move), rewriting only the band
Y; the flanking descent-X and port-Y keep the handlers' order, so the band read
outer-to-inner one way and the descent the other and the two crossed at the corner
where the band turned down (issue #1835).

The full map is inlined rather than committed as a fixture: it also trips the
unrelated symmetric-diamond centreline abort under ``validate=True`` (#1836), which
would red every corpus invariant that renders it.  ``check_merge_confluence_band_order``
resolves each peel-off tail's port through the merge chain and flags a co-travelling
distinct-line pair that ranks one way on the band and the other on the descent.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from nf_metro.layout.engine import compute_layout
from nf_metro.layout.routing import compute_station_offsets, route_edges_centred
from nf_metro.layout.routing.invariants import check_merge_confluence_band_order
from nf_metro.parser.mermaid import parse_metro_mermaid

REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLES = REPO_ROOT / "examples"
PORT = "orf_calling__entry_left_7"

RIBOSEQ_CONFLUENCE = r"""%%metro title: nf-core/riboseq
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
        equalise[Equalise\nread lengths]

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
        genomecov[BEDTools\ngenomecov]
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
        hybrid_merge[Merge &\nfilter GTF]

        stringtie -->|rnaseq| gffcompare
        gffcompare -->|rnaseq| hybrid_merge
        hybrid_merge -->|rnaseq| hybrid_gtf_out
    end


    subgraph orf_calling [ORF discovery & calling]
        star_hybrid[STAR:\nhybrid 2nd pass]
        ribotish[Ribo-TISH]
        ribocode[RiboCode]
        ribotricer[Ribotricer]
        rpbp[Rp-Bp]
        price[PRICE]
        orf_merge[Merge ORF\ncatalogue]

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
        plastid_psite[plastid\nP-site]
        plastid_wiggle[plastid\nwiggle]
        quantify_orf_psite[Quantify ORF\nP-sites]
        psite_counts_gene[Gene in-frame\nP-sites]

        ribowaltz -->|riboseq| quantify_orf_psite
        plastid_psite -->|riboseq| plastid_wiggle
        plastid_wiggle -->|riboseq| quantify_orf_psite
        plastid_wiggle -->|riboseq| psite_counts_gene
    end

    subgraph te [Translational efficiency]
        te_prep_gene[Gene count\nmatrix]
        te_prep_orf[ORF count\nmatrix]
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
    anota2seq -->|riboseq,rnaseq| multiqc_final
    hybrid_merge -->|annotation| star_hybrid
    hybrid_merge -->|annotation| ribotish
    hybrid_merge -->|annotation| ribotricer
    hybrid_merge -->|annotation| ribocode
"""


def _route(text: str):
    graph = parse_metro_mermaid(text)
    compute_layout(graph)
    offsets = compute_station_offsets(graph)
    routes = route_edges_centred(graph, station_offsets=offsets)
    return graph, routes


def test_riboseq_confluence_routes_without_band_descent_crossing() -> None:
    graph, routes = _route(RIBOSEQ_CONFLUENCE)
    reaching = {
        rp.line_id
        for rp in routes
        if rp.is_inter_section
        and (rp.edge.target == PORT or rp.edge.target.startswith("__merge"))
    }
    assert {"annotation", "riboseq"} <= reaching
    assert check_merge_confluence_band_order(routes, graph) == []


def test_check_catches_a_planted_confluence_crossing() -> None:
    graph, routes = _route(RIBOSEQ_CONFLUENCE)
    annotation = next(
        rp for rp in routes if rp.line_id == "annotation" and rp.edge.target == PORT
    )
    riboseq = next(
        rp
        for rp in routes
        if rp.line_id == "riboseq" and rp.edge.target.startswith("__merge")
    )
    # Re-seat the exempt annotation descent inboard of the riboseq column, onto the
    # port Y riboseq holds, recreating the pre-fix crossing at the band's turn.
    rib_peel_x = riboseq.points[-3][0]
    rib_port_y = riboseq.points[-1][1]
    pts = list(annotation.points)
    pts[-3] = (rib_peel_x + 4.0, pts[-3][1])
    pts[-2] = (rib_peel_x + 4.0, rib_port_y - 4.0)
    pts[-1] = (pts[-1][0], rib_port_y - 4.0)
    annotation.points = pts
    violations = check_merge_confluence_band_order(routes, graph)
    assert any(
        {v.line_a, v.line_b} == {"annotation", "riboseq"} and v.port_id == PORT
        for v in violations
    )


def _corpus_fixtures() -> list[Path]:
    paths: list[Path] = []
    paths.extend(sorted(EXAMPLES.glob("*.mmd")))
    paths.extend(sorted((EXAMPLES / "topologies").glob("*.mmd")))
    return paths


@pytest.mark.parametrize("fixture", _corpus_fixtures(), ids=lambda p: p.stem)
def test_no_shipped_fixture_trips_the_confluence_oracle(fixture: Path) -> None:
    graph, routes = _route(fixture.read_text())
    assert check_merge_confluence_band_order(routes, graph) == []
