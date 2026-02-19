#!/usr/bin/env nextflow

// Pipeline with subworkflows: preprocessing -> alignment -> quantification

params.reads = "reads/*_{1,2}.fastq.gz"
params.genome = "genome.fa"
params.gtf = "genes.gtf"

process FASTQC {
    input:
    tuple val(sample_id), path(reads)

    output:
    path("*.zip"), emit: zip

    script:
    """
    touch ${sample_id}_fastqc.zip
    """
}

process TRIMGALORE {
    input:
    tuple val(sample_id), path(reads)

    output:
    tuple val(sample_id), path("*_trimmed.fq.gz"), emit: reads
    path("*_report.txt"), emit: log

    script:
    """
    touch ${sample_id}_1_trimmed.fq.gz ${sample_id}_2_trimmed.fq.gz
    touch ${sample_id}_trimming_report.txt
    """
}

process STAR_GENOMEGENERATE {
    input:
    path(genome)
    path(gtf)

    output:
    path("star_index"), emit: index

    script:
    """
    mkdir star_index
    touch star_index/SA
    """
}

process STAR_ALIGN {
    input:
    tuple val(sample_id), path(reads)
    path(index)

    output:
    tuple val(sample_id), path("*.bam"), emit: bam
    path("*.Log.final.out"), emit: log

    script:
    """
    touch ${sample_id}.Aligned.sortedByCoord.out.bam
    touch ${sample_id}.Log.final.out
    """
}

process SAMTOOLS_SORT {
    input:
    tuple val(sample_id), path(bam)

    output:
    tuple val(sample_id), path("*.sorted.bam"), emit: bam

    script:
    """
    touch ${sample_id}.sorted.bam
    """
}

process SAMTOOLS_INDEX {
    input:
    tuple val(sample_id), path(bam)

    output:
    tuple val(sample_id), path("*.bai"), emit: bai

    script:
    """
    touch ${sample_id}.sorted.bam.bai
    """
}

process SALMON_QUANT {
    input:
    tuple val(sample_id), path(bam)
    path(gtf)

    output:
    path("${sample_id}_quant"), emit: results

    script:
    """
    mkdir ${sample_id}_quant
    touch ${sample_id}_quant/quant.sf
    """
}

process MULTIQC {
    input:
    path(reports)

    output:
    path("multiqc_report.html")

    script:
    """
    touch multiqc_report.html
    """
}

workflow PREPROCESS {
    take:
    reads

    main:
    FASTQC(reads)
    TRIMGALORE(reads)

    emit:
    reads = TRIMGALORE.out.reads
    fastqc_zip = FASTQC.out.zip
    trim_log = TRIMGALORE.out.log
}

workflow ALIGNMENT {
    take:
    reads
    genome
    gtf

    main:
    STAR_GENOMEGENERATE(genome, gtf)
    STAR_ALIGN(reads, STAR_GENOMEGENERATE.out.index.collect())
    SAMTOOLS_SORT(STAR_ALIGN.out.bam)
    SAMTOOLS_INDEX(SAMTOOLS_SORT.out.bam)

    emit:
    bam = SAMTOOLS_SORT.out.bam
    bai = SAMTOOLS_INDEX.out.bai
    star_log = STAR_ALIGN.out.log
}

workflow QUANTIFICATION {
    take:
    bam
    gtf

    main:
    SALMON_QUANT(bam, gtf)

    emit:
    results = SALMON_QUANT.out.results
}

workflow {
    reads_ch = Channel.of(
        ["sample1", [file("reads/s1_1.fq.gz"), file("reads/s1_2.fq.gz")]]
    )
    genome_ch = Channel.of(file("genome.fa"))
    gtf_ch = Channel.of(file("genes.gtf"))

    PREPROCESS(reads_ch)
    ALIGNMENT(PREPROCESS.out.reads, genome_ch, gtf_ch)
    QUANTIFICATION(ALIGNMENT.out.bam, gtf_ch)

    MULTIQC(
        PREPROCESS.out.fastqc_zip
            .mix(PREPROCESS.out.trim_log)
            .mix(ALIGNMENT.out.star_log)
            .mix(QUANTIFICATION.out.results)
            .collect()
    )
}
