#!/usr/bin/env nextflow

// A simple flat pipeline: FASTQC + trim + align + sort + multiqc

params.reads = "reads/*_{1,2}.fastq.gz"
params.reference = "genome.fa"

process FASTQC {
    input:
    tuple val(sample_id), path(reads)

    output:
    path("*.html"), emit: html
    path("*.zip"), emit: zip

    script:
    """
    echo fastqc ${reads}
    touch ${sample_id}_fastqc.html ${sample_id}_fastqc.zip
    """
}

process TRIM_READS {
    input:
    tuple val(sample_id), path(reads)

    output:
    tuple val(sample_id), path("*_trimmed.fastq.gz")

    script:
    """
    echo trim ${reads}
    touch ${sample_id}_1_trimmed.fastq.gz ${sample_id}_2_trimmed.fastq.gz
    """
}

process ALIGN {
    input:
    tuple val(sample_id), path(reads)
    path(reference)

    output:
    tuple val(sample_id), path("*.bam")

    script:
    """
    echo align ${reads} to ${reference}
    touch ${sample_id}.bam
    """
}

process SORT_BAM {
    input:
    tuple val(sample_id), path(bam)

    output:
    tuple val(sample_id), path("*.sorted.bam")

    script:
    """
    echo sort ${bam}
    touch ${sample_id}.sorted.bam
    """
}

process MULTIQC {
    input:
    path(reports)

    output:
    path("multiqc_report.html")

    script:
    """
    echo multiqc .
    touch multiqc_report.html
    """
}

workflow {
    reads_ch = Channel.of(
        ["sample1", [file("reads/s1_1.fq.gz"), file("reads/s1_2.fq.gz")]]
    )
    reference_ch = Channel.of(file("genome.fa"))

    FASTQC(reads_ch)
    TRIM_READS(reads_ch)
    ALIGN(TRIM_READS.out, reference_ch.collect())
    SORT_BAM(ALIGN.out)
    MULTIQC(
        FASTQC.out.zip
            .mix(SORT_BAM.out.map { it[1] })
            .collect()
    )
}
