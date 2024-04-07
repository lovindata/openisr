import { ProcessRadioOptions } from "@/features/processes/components/organisms/ProcessRadioOrg/ProcessRadioOptions";
import { HorizontalRadioMol } from "@/features/shared/components/molecules/HorizontalRadioMol";

interface Props {
  value: ProcessRadioOptions;
  setValue: (value: ProcessRadioOptions) => void;
}

export function ProcessRadioOrg({ value, setValue }: Props) {
  const values = Object.values(ProcessRadioOptions);
  return (
    <HorizontalRadioMol
      possibleValues={values}
      value={value}
      setValue={setValue}
      className="w-72"
    />
  );
}
